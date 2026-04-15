/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <atomic>
#include <cstdlib>
#include <limits>
#include <list>
#include <shared_mutex>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <Tensile/Comparison.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/SolutionLibrary.hpp>

#include <Tensile/AMDGPU_Detail.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/TensorDescriptor_Detail.hpp>

namespace TensileLite
{
    /**
     * Sentinel: unbounded cache (no LRU eviction). Not a valid LRU cap.
     */
    inline constexpr size_t CacheMapUnlimitedEntries = std::numeric_limits<size_t>::max();

    /**
     * Read cache cap from HIPBLASLT_CACHE_MAP_MAX_ENTRIES or
     * TENSILE_CACHE_MAP_MAX_ENTRIES (HIP name wins if both set).
     *
     * When unset: unlimited entries (same as CacheMapUnlimitedEntries).
     *
     * Parsed as signed base-10:
     *      0: disable caching (find always misses, add is no-op)
     *     -1: unlimited (any negative is treated the same)
     *  N > 0: LRU cap of N leaf entries
     */
    inline size_t cacheMapMaxLeafEntriesFromEnv()
    {
        static const size_t limit = []() -> size_t {
            const char* e = std::getenv("HIPBLASLT_CACHE_MAP_MAX_ENTRIES");

            if(!e || *e == '\0')
            {
                e = std::getenv("TENSILE_CACHE_MAP_MAX_ENTRIES");
            }

            if(e && *e != '\0')
            {
                char*     end = nullptr;
                long long v   = std::strtoll(e, &end, 10);

                if(end != e)
                {
                    if(v == 0)
                        return 0;
                    if(v < 0)
                        return CacheMapUnlimitedEntries;
                    return static_cast<size_t>(v);
                }
            }

            return CacheMapUnlimitedEntries;
        }();

        return limit;
    }

    template <typename Head, typename... Tail>
    struct ReversedKeyTuple
    {
        using type = decltype(std::tuple_cat(
            std::declval<typename ReversedKeyTuple<Tail...>::type const&>(),
            std::declval<std::tuple<Head> const&>()));
    };

    template <typename Single>
    struct ReversedKeyTuple<Single>
    {
        using type = std::tuple<Single>;
    };

    template <typename KeyTuple>
    struct CacheKeyTupleHash;

    template <typename... Ts>
    struct CacheKeyTupleHash<std::tuple<Ts...>>
    {
        size_t operator()(std::tuple<Ts...> const& t) const noexcept
        {
            return std::apply(
                [](auto const&... args) { return hash_combine(args...); }, t);
        }
    };

    /**
     * Thread-safe flat cache with optional LRU eviction on the number of leaf
     * entries. Keys are passed to find() / add() in the same order as before
     * (opposite of the template parameter list); see legacy CacheMap comment.
     *
     * maxLeafEntries:
     *   0: caching disabled (find always returns nullValue; add does nothing)
     *   CacheMapUnlimitedEntries: no eviction (unbounded)
     *   otherwise: LRU cap of maxLeafEntries entries
     */
    template <typename Value, typename... Keys>
    class CacheMap
    {
        using KeyTuple = typename ReversedKeyTuple<Keys...>::type;
        using Hash     = CacheKeyTupleHash<KeyTuple>;

    public:
        CacheMap(Value const& nullValue, size_t maxLeafEntries = CacheMapUnlimitedEntries)
            : m_nullValue(nullValue)
            , m_maxLeafEntries(maxLeafEntries)
            , m_lookupEfficiency(Debug::Instance().printLookupEfficiency())
            , m_lookups(0)
            , m_hits(0)
        {
        }

        ~CacheMap()
        {
            if(m_lookupEfficiency)
                std::cout << "CacheMap: " << m_hits << "/" << m_lookups << " cache hits"
                          << std::endl;
        }

        template <typename... Ks>
        Value find(Ks const&... keys)
        {
            if(m_maxLeafEntries == 0)
            {
                if(m_lookupEfficiency)
                    m_lookups++;
                return m_nullValue;
            }

            KeyTuple const key(keys...);

            if(m_maxLeafEntries == CacheMapUnlimitedEntries)
            {
                std::shared_lock<std::shared_timed_mutex> lock(m_mutex);
                auto                                      rv = findInMapUnlocked(key);

                if(m_lookupEfficiency)
                {
                    m_lookups++;
                    if(rv != m_nullValue)
                        m_hits++;
                }
                return rv;
            }

            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);
            auto                                     it = m_map.find(key);
            if(it == m_map.end())
            {
                if(m_lookupEfficiency)
                {
                    m_lookups++;
                }
                return m_nullValue;
            }

            touchUnlocked(key);

            if(m_lookupEfficiency)
            {
                m_lookups++;
                if(it->second != m_nullValue)
                    m_hits++;
            }

            return it->second;
        }

        template <typename... Ks>
        void add(Value const& value, Ks const&... ks)
        {
            if(m_maxLeafEntries == 0)
                return;

            KeyTuple key(ks...);

            std::lock_guard<std::shared_timed_mutex> lock(m_mutex);

            if(m_maxLeafEntries == CacheMapUnlimitedEntries)
            {
                m_map.insert_or_assign(key, value);
                return;
            }

            auto it = m_map.find(key);
            if(it != m_map.end())
            {
                it->second = value;
                touchUnlocked(key);
                return;
            }

            while(m_map.size() >= m_maxLeafEntries)
                evictLruUnlocked();

            m_map.emplace(key, value);
            m_lru.push_back(key);
            m_lruPos[key] = std::prev(m_lru.end());
        }

    private:
        Value findInMapUnlocked(KeyTuple const& key)
        {
            auto it = m_map.find(key);
            if(it == m_map.end())
                return m_nullValue;
            return it->second;
        }

        void touchUnlocked(KeyTuple const& key)
        {
            auto i = m_lruPos.find(key);
            if(i != m_lruPos.end())
                m_lru.splice(m_lru.end(), m_lru, i->second);
        }

        void evictLruUnlocked()
        {
            if(m_lru.empty())
                return;
            KeyTuple victim = m_lru.front();
            m_lru.pop_front();
            m_lruPos.erase(victim);
            m_map.erase(victim);
        }

        std::unordered_map<KeyTuple, Value, Hash> m_map;
        std::list<KeyTuple>                       m_lru;
        std::unordered_map<KeyTuple, typename std::list<KeyTuple>::iterator, Hash> m_lruPos;

        std::shared_timed_mutex m_mutex;
        Value                   m_nullValue;
        size_t                  m_maxLeafEntries;

        bool                 m_lookupEfficiency;
        std::atomic<int64_t> m_lookups;
        std::atomic<int64_t> m_hits;
    };

    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    class CachingLibrary : public SolutionLibrary<MyProblem, MySolution>
    {
    public:
        using Library = SolutionLibrary<MyProblem, MySolution>;
        using Cache  = CacheMap<std::tuple<std::shared_ptr<MySolution>, double>, AMDGPU, MyProblem>;
        using Caches = CacheMap<SolutionVector<MySolution>, AMDGPU, MyProblem>;
        using CachesAllSolsFlag
            = CacheMap<bool, AMDGPU, MyProblem>;
        using CachesGroupedGemm
            = CacheMap<SolutionVector<MySolution>, AMDGPU, std::vector<MyProblem>>;

        CachingLibrary(std::shared_ptr<Library> subLibrary)
            : m_subLibrary(subLibrary)
            , m_cache(std::make_tuple(nullptr, std::numeric_limits<double>::max()),
                      cacheMapMaxLeafEntriesFromEnv())
            , m_caches(SolutionVector<MySolution>{}, cacheMapMaxLeafEntriesFromEnv())
            , m_cachesAllSolutions(false, cacheMapMaxLeafEntriesFromEnv())
            , m_cachesGroupedGemm(SolutionVector<MySolution>{},
                                  cacheMapMaxLeafEntriesFromEnv())
        {
        }

        virtual std::shared_ptr<MySolution> getSolutionByIndex(MyProblem const& problem,
                                                               Hardware const&  hardware,
                                                               const int index) const override
        {
            return m_subLibrary->getSolutionByIndex(problem, hardware, index);
        }

        virtual std::shared_ptr<MySolution> findBestSolution(MyProblem const& problem,
                                                             Hardware const&  hardware,
                                                             double*          fitness
                                                             = nullptr) const override
        {
            try
            {
                double cachedFitness = std::numeric_limits<double>::max();
                fitness              = (fitness) ? fitness : &cachedFitness;

                auto const&                 amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                std::shared_ptr<MySolution> solution;
                std::tie(solution, *fitness) = m_cache.find(problem, amdgpu);

                if(solution)
                    return solution;

                solution = m_subLibrary->findBestSolution(problem, hardware, fitness);
                if(solution)
                    m_cache.add(std::make_tuple(solution, *fitness), problem, amdgpu);

                return solution;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findBestSolution(problem, hardware, fitness);
            }
        }

        virtual SolutionSet<MySolution>
            findAllSolutions(MyProblem const&          problem,
                             Hardware const&           hardware,
                             SolutionLibrarySearchType searchType
                             = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_subLibrary->findAllSolutions(problem, hardware, searchType);
        }

        virtual SolutionSet<MySolution>
            findAllSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        SolutionLibrarySearchType     searchType
                                        = SolutionLibrarySearchType::DEFAULT) const override
        {
            return m_subLibrary->findAllSolutionsGroupedGemm(problems, hardware, searchType);
        }

        std::shared_ptr<MySolution> findSolutionInCache(MyProblem const& problem,
                                                        Hardware const&  hardware) const
        {
            auto const& amdgpu = dynamic_cast<AMDGPU const&>(hardware);

            return std::get<std::shared_ptr<MySolution>>(m_cache.find(problem, amdgpu));
        }

        virtual std::string type() const override
        {
            return "Caching Library";
        }
        virtual std::string description() const override
        {
            return "Caching Library";
        }

        std::shared_ptr<Library> library() const
        {
            return m_subLibrary;
        }

        virtual SolutionVector<MySolution> findTopSolutions(MyProblem const& problem,
                                                            Hardware const&  hardware,
                                                            int numSolutions) const override
        {
            try
            {
                auto const&                amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                SolutionVector<MySolution> solutions;
                bool                       cacheAlreadyContainAll;
                solutions = m_caches.find(problem, amdgpu);
                cacheAlreadyContainAll = m_cachesAllSolutions.find(problem, amdgpu);
                // set flag in case of early return
                lastFindTopRetAll = cacheAlreadyContainAll;

                if(solutions.size() >= numSolutions || cacheAlreadyContainAll)
                    return solutions;

                solutions = m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
                if(solutions.size() != 0)
                {
                    bool alreadyRetAll = m_subLibrary->lastFindTopAlreadyRetAll();
                    m_caches.add(solutions, problem, amdgpu);
                    m_cachesAllSolutions.add(alreadyRetAll, problem, amdgpu);
                    // debug
                    // std::cout << "m_cachesAllSolutions.add() with solution.size() = " << solutions.size()
                    //           << " and alreadyRetAll: " << (alreadyRetAll? "True" : "False") << std::endl;
                }

                // can't reach the requested number, means findTop already done its best
                lastFindTopRetAll = (solutions.size() < numSolutions);
                return solutions;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
            }
            // TODO- redundant ??
            return m_subLibrary->findTopSolutions(problem, hardware, numSolutions);
        }

        virtual bool lastFindTopAlreadyRetAll() const override
        {
            return lastFindTopRetAll;
        }

        virtual SolutionVector<MySolution>
            findTopSolutionsGroupedGemm(std::vector<MyProblem> const& problems,
                                        Hardware const&               hardware,
                                        int                           numSolutions) const override
        {
            try
            {
                auto const&                amdgpu = dynamic_cast<AMDGPU const&>(hardware);
                SolutionVector<MySolution> solutions;
                solutions = m_cachesGroupedGemm.find(problems, amdgpu);

                if(solutions.size() != 0)
                    return solutions;

                solutions
                    = m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
                if(solutions.size() != 0)
                    m_cachesGroupedGemm.add(solutions, problems, amdgpu);

                return solutions;
            }
            catch(std::bad_cast const& exc)
            {
                return m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
            }
            return m_subLibrary->findTopSolutionsGroupedGemm(problems, hardware, numSolutions);
        }

    private:
        std::shared_ptr<Library>  m_subLibrary;
        mutable Cache             m_cache;
        mutable Caches            m_caches;
        mutable CachesAllSolsFlag m_cachesAllSolutions;
        mutable CachesGroupedGemm m_cachesGroupedGemm;
        mutable std::atomic<bool> lastFindTopRetAll = false;
    };

#if 0
    struct ContractionCachingLibrary: public CachingLibrary<ContractionProblemGemm>
    {
        using Super = CachingLibrary<ContractionProblemGemm>;
        using Library = typename Super::Library;
        using Key = typename Super::Key;

        ContractionCachingLibrary(std::shared_ptr<Library> subLibrary)
            : CachingLibrary<ContractionProblemGemm>(subLibrary)
        {}

    };
#endif

} // namespace TensileLite
