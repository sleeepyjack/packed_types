#include <iostream>
#include "packed_types.cuh"

int main (int argc, char *argv[])
{
    // payload of packed data type
    using base_t = std::uint64_t;

    // packed type with three fields: 13 bits, 8 bits, 34 bits, (+ 9 bits padding)
    using pack_t = PackedTriple<base_t, 13, 8, 34>;
    // or using pack_t = PackedPair<base_t, _, _>;
    // or using pack_t = PackedQuadruple<base_t, _, _, _, _>;
    
    std::cout 
        << "size of pack_t="  
        << sizeof(base_t)
        << " is equal to the size of base_t="
        << sizeof(base_t)
        << std::endl;
        
    // print data layout (curly braces denote padding bits)
    std::cout 
        << "bit partition: [[" 
        << pack_t::first_bits() << "]["
        << pack_t::second_bits() << "]["
        << pack_t::third_bits() << "]{"
        << pack_t::padding_bits() << "}]"
        << std::endl; 

    // build packed triple
    pack_t triple{1234, 12, 123};

    // get fields
    std::cout 
        << "triple = (" 
        << triple.first() 
        << ", " 
        << triple.second() 
        << ", " 
        << triple.third() 
        << ")" 
        << std::endl;

    // also supports std::get<>
    std::cout 
        << "std::get<> triple = (" 
        << get<0>(triple) 
        << ", " 
        << get<1>(triple) 
        << ", " 
        << get<2>(triple) 
        << ")" 
        << std::endl;

    // and member functions with similar syntax i.e. get<>()/set<>()
    // NOTE you may need to provide additional info to the compiler 
    // e.g. "triple.template get<0>()"
    std::cout 
        << "triple.get<> triple = (" 
        << triple.get<0>() 
        << ", " 
        << triple.get<1>() 
        << ", " 
        << triple.get<2>() 
        << ")" 
        << std::endl;

    // update third field
    triple.third(42);
    // or triple.set<2>(42) or triple.template set<2>(42)
    std::cout << "third = " << triple.third() << std::endl;

    // following line should trigger an assertion error since 12345 needs more than 8 bit
    // triple.second(12345);
    std::cout 
        << std::boolalpha
        << "should be false: " 
        << pack_t::is_valid_second(12345) 
        << std::endl;

    std::cout 
        << "maximum value for second field is " 
        << pack_t::second_max() 
        << std::endl;

    // support for atomic updates:
    // comes with specializations for CUDA's atomicCAS() and atomicExch()
    // also valid: std::atomic<pack_t> 
}
