#include <filesystem>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>
#include <unordered_map>

namespace test_utils
{

class TestController
{
    // TODO: update CMakeLists.txt so this installs properly
    inline const static std::filesystem::path control_path = "../test/control.txt";
public:
    TestController* get_instance()
    {
        if (!TestController::instance)
            TestController::instance = new TestController();
        
        return TestController::instance;
    }

    TestController(const TestController&) = delete;
    TestController& operator=(const TestController&) = delete;

    ~TestController()
    {
        if (TestController::instance)
        {
            delete TestController::instance;
            TestController::instance = nullptr;
        }
    }

    void skip_if_disabled()
    {
        // Get the fully qualified name of the currently running test,
        // in the format: "<suite name>.<test name>"
        std::stringstream ss;
        ss << ::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()
           << "."
           << ::testing::UnitTest::GetInstance()->current_test_info()->name();
        std::string qualified_name = ss.str();

        
    }
    
private:
    TestController()
    {
        TestController::gfx_id = TestController::get_arch();
        TestController::parse_control_data();
    }

    inline static std::string get_arch()
    {
        int device_id = test_common_utils::obtain_device_from_ctest();
        hipDeviceProp_t dev_prop;
        HIP_CHECK(hipGetDeviceProperties(&dev_prop, device_id));
        const char* arch_name = dev_prop.gcnArchName;
        return std::string(arch_name);

        // static constexpr auto length = sizeof(hipDeviceProp_t::gcnArchName);
        // const char* arch_end = std::find_if(arch_name,
        //                                     arch_name + length,
        //                                     [](const char& val) { return val == ':' || val == '\0'; });
    }
    
    inline static bool parse_line(std::string& line, std::regex& test_regex, std::regex& arch_regex)
    {
        const std::string delimiter = "=";
        std::string::size_type split_index = line.find(delimiter);
        if (split_index == std::string::npos)
        {
            std::cout << "Error parsing line. Unable to find '='." << std::endl;
            return false;
        }
            
        std::string lhs = line.substr(0, split_index);
        std::string rhs = line.substr(split_index + 1, line.size());
        
        // Capture group to get all non-whitespace between the '/' ... '/'
        std::regex lhs_regex("^\\s*\\/(.+)\\/\\s*$");
        std::smatch match_result;
        if (std::regex_match(lhs, match_result, lhs_regex))
        {
            if (match_result.size() == 2)
                test_regex = std::regex(match_result[1].str());
            else
            {
                std::cout << "Error parsing regex: \"" << lhs << "\"" << std::endl;
                return false;
            }
        }

        // Check if rhs is a regex
        std::regex rhs_regex("^\\s*\\/(.+)\\/\\s*$");
        if (std::regex_match(rhs, match_result, rhs_regex))
        {
            if (match_result.size() == 2)
                arch_regex = std::regex(match_result[1].str());
            else
            {
                std::cout << "Error parsing regex: \"" << rhs << "\"" << std::endl;
                return false;
            }
        }
        // Otherwise, it should be keywords
        else
        {
            // String should consist of keyword(s), and optionally '|' characters, and/or whitespace
            std::stringstream ss;
            std::regex re_delim("\\s*\\|\\s*");
            std::sregex_token_iterator iter(rhs.begin(), rhs.end(), re_delim, -1);
            const std::sregex_token_iterator end; // default constructor creates an "end of sequence" iterator
            while (iter != end)
            {
                auto kwd_it = TestController::keywords.find(*iter);
                if (kwd_it == TestController::keywords.end())
                {
                    std::cout << "Error: unrecognized keyword: \"" << *iter << "\"" << std::endl;
                    return false;
                }
                else
                {
                    ss << kwd_it->second;
                }
                
                iter++;
                if (iter != end)
                    ss << "|";
            }
            arch_regex = std::regex(ss.str());
        }
        
        return true;
    }
    
    inline static void parse_control_data()
    {
        std::ifstream control_file(control_path.c_str());

        if (!control_file)
        {
            std::cout << "Error: Cannot open control file at: \"" << TestController::control_path.c_str() << "\"" << std::endl;
            return;
        }

        std::string line;
        while (std::getline(control_file, line))
        {
            std::regex test_regex;
            std::regex arch_regex;
            TestController::parse_line(line, test_regex, arch_regex);
        }
        
        control_file.close();
    }

    // Pointer to the singleton instance.
    inline static TestController* instance = nullptr;

    inline static std::string gfx_id;
    
    // Maps <test name regex> => <gfx id regex>, keywords are transformed into a regex before being stored here
    inline static std::unordered_map<std::string, std::string> control_data;
    
    // Maps <keyword> => <regex string for all gfx ids that keyword represents>
    const inline static std::unordered_map<std::string, std::string> keywords = {
        {"all",           "(gfx[0-9a-f]+)"},
        {"apus",          "(gfx1103|gfx1150|gfx1151|gfx1152)"},
        {"navi2x-family", "(gfx1030|gfx1031|gfx1032)"},
        {"navi3x-family", "(gfx1100|gfx1101|gfx1102)"},
        {"navi4x-family", "(gfx1200|gfx1201)"},
        {"mi100-family",  "(gfx908)"},
        {"mi200-family",  "(gfx90a)"},
        {"mi300-family",  "(gfx942|gfx950)"}
    };

};

}
