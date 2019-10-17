#ifndef PACKED_TYPES_CUH
#define PACKED_TYPES_CUH

#include <cstdint>
#include <type_traits>
#include <assert.h>
#include "cudahelpers/cuda_helpers.cuh"

// INFO you can find the actual types as using statements at the end of this file

// bit-wise reinterpret one fundamental type as another fundamental type
template<class To, class From>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr To reinterpret_as(From from) noexcept
{
    static_assert(
        std::is_fundamental<To>::value, 
        "Target type must be fundamental.");

    static_assert(
        std::is_fundamental<From>::value, 
        "Input type must be fundamental.");

    union reinterpreter_t
    {
        From from;
        To to;

        HOSTDEVICEQUALIFIER
        constexpr reinterpreter_t() noexcept : to(To()) {}
    } reinterpreter;
    
    reinterpreter.from = from;
    return reinterpreter.to;
}

namespace detail
{

template<
    class Base,
    Base  FirstBits,
    Base  SecondBits,
    Base  ThirdBits = 0,
    Base  FourthBits = 0>
class Pack
{
    // memory layout: MSB->padding|fourth|third|second|first<-LSB

    static_assert(
        FirstBits != 0 && SecondBits != 0,
        "FirstBits and SecondBits both may not be zero.");

    static_assert(
        !(ThirdBits == 0 && FourthBits != 0),
        "Third type cannot be zero-width if fourth type has non-zero width.");

    static_assert(
        FirstBits + SecondBits + ThirdBits + FourthBits <= sizeof(Base) * 8,
        "Too many bits for chosen datatype.");

    static_assert(
        std::is_fundamental<Base>::value,
        "Base type must be fundamental.");

    // leftover bits are padding
    static constexpr Base PaddingBits = 
        (sizeof(Base) * 8) - (FirstBits + SecondBits + ThirdBits + FourthBits);

    // bit masks for each individual field
    static constexpr Base first_mask = ((Base{1} << FirstBits) - Base{1});

    static constexpr Base second_mask = 
        ((Base{1} << SecondBits) - Base{1}) << 
            (FirstBits);

    static constexpr Base third_mask = 
        (ThirdBits == 0) ? 
            Base{0} : 
            ((Base{1} << ThirdBits) - Base{1}) << 
                (FirstBits + SecondBits);

    static constexpr Base fourth_mask = 
        (FourthBits == 0) ? 
            Base{0} : 
            ((Base{1} << FourthBits) - Base{1}) << 
                (FirstBits + SecondBits + ThirdBits);

    static constexpr Base padding_mask = 
        (PaddingBits == 0) ? 
            Base{0} : 
            ((Base{1} << PaddingBits) - Base{1}) << 
                (FirstBits + SecondBits + ThirdBits + FourthBits);


public:
    using base_type = Base;

    // number of bits per field
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base padding_bits() noexcept { return PaddingBits; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base first_bits() noexcept { return FirstBits; }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base second_bits() noexcept { return SecondBits; }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base third_bits() noexcept { return ThirdBits; }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base fourth_bits() noexcept { return FourthBits; }

    // CONSTRUCTORS
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack() noexcept : base_{empty().base_} {}

    template<
        class FirstType,
        class SecondType,
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
	    class   = std::enable_if_t<B1 == 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER    
    constexpr explicit Pack(
        FirstType first_, 
        SecondType second_) noexcept : base_{empty().base_} 
    {
        first(first_);
        second(second_);
    }

    template<
        class FirstType,
        class SecondType,
        class ThirdType,
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
        class   = std::enable_if_t<B1 != 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack(
        FirstType first_, 
        SecondType second_, 
        ThirdType third_) noexcept : base_{empty().base_} 
    {
        first(first_);
        second(second_);
        third(third_);
    }

    template<
        class FirstType,
        class SecondType,
        class ThirdType,
        class FourthType,
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
	    class   = std::enable_if_t<B1 != 0 && B2 != 0>>
    HOSTDEVICEQUALIFIER
    constexpr explicit Pack(
        FirstType first_, 
        SecondType second_, 
        ThirdType third_, 
        FourthType fourth_) noexcept : base_{empty().base_} 
    {
        first(first_);
        second(second_);
        third(third_);
        fourth(fourth_);
    }

    constexpr Pack(const Pack&) noexcept = default;
    constexpr Pack(Pack&& pair) noexcept = default;

    // returns an empty pack
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Pack empty() noexcept 
    { 
        return Pack(Base{0});
    }

    // SETTERS
    // by field name
    template<class First>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(First first_) noexcept
    {
        debug_printf(
            "Type reinterpretation inside setter of first field.");
        first(reinterpret_as<Base>(first_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(Base first_) noexcept
    {
        assert(is_valid_first(first_));
        base_ = (base_ & ~first_mask) + (first_ & first_mask);
    }

    template<class Second>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(Second second_) noexcept
    {
        debug_printf(
            "Type reinterpretation inside setter of second field.");
        second(reinterpret_as<Base>(second_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(Base second_) noexcept
    {
        assert(is_valid_second(second_));
        constexpr auto shift = FirstBits;
        base_ = (base_ & ~second_mask) + ((second_ << shift) & second_mask);
    }

    template<
        class Third,
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void third(Third third_) noexcept
    {
        debug_printf(
            "Type reinterpretation inside setter of third field.");
        third(reinterpret_as<Base>(third_));
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void third(Base third_) noexcept
    {
        assert(is_valid_third(third_));
        constexpr auto shift = FirstBits + SecondBits;
        base_ = (base_ & ~third_mask) + ((third_ << shift) & third_mask);
    }

    template<
        class Fourth,
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void fourth(Fourth fourth_) noexcept
    {
        debug_printf(
            "Type reinterpretation inside setter of fourth field.");
        fourth(reinterpret_as<Base>(fourth_));
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void fourth(Base fourth_) noexcept
    {
        assert(is_valid_fourth(fourth_));
        constexpr auto shift = FirstBits + SecondBits + ThirdBits;
        base_ = (base_ & ~fourth_mask) + ((fourth_ << shift) & fourth_mask);
    }

    // GETTERS
    // by field name
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T first() const noexcept
    {
        debug_printf(
            "Type reinterpretation inside getter of first field.");
        return reinterpret_as<T>(first());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base first() const noexcept
    {
        return (base_ & first_mask);
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T second() const noexcept
    {
        debug_printf(
            "Type reinterpretation inside getter of second field.");
        return reinterpret_as<T>(second());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base second() const noexcept
    {
        return ((base_ & second_mask) >> (FirstBits));
    }

    template<
        class T,
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T third() const noexcept
    {
        debug_printf(
            "Type reinterpretation inside getter of third field.");
        return reinterpret_as<T>(third());
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base third() const noexcept
    {
        return ((base_ & third_mask) >> (FirstBits + SecondBits));
    }

    template<
        class T,
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr T fourth() const noexcept
    {
        debug_printf(
            "Type reinterpretation inside getter of fourth field.");
        return reinterpret_as<T>(fourth());
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base fourth() const noexcept
    {
        return ((base_ & fourth_mask) >> (FirstBits + SecondBits + ThirdBits));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base base() const noexcept
    {
        return (base_ & ~padding_mask);
    }

    // SETTERS
    // set<index>(value)
    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, void> 
    set(T first_) noexcept { first<T>(first_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, void> 
    set(Base first_) noexcept { first(first_); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, void> 
    set(T second_) noexcept { second<T>(second_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, void> 
    set(Base second_) noexcept { second(second_); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, void> 
    set(T third_) noexcept { third<T>(third_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, void> 
    set(Base third_) noexcept { third(third_); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, void> 
    set(T fourth_) noexcept { fourth<T>(fourth_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, void> 
    set(Base fourth_) noexcept { fourth(fourth_); }

    // GETTERS
    // get<index>()
    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, T> 
    get() const noexcept { return first<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, Base> 
    get() const noexcept { return first(); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, T> 
    get() const noexcept { return second<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, Base> 
    get() const noexcept { return second(); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, T> 
    get() const noexcept { return third<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, Base> 
    get() const noexcept { return third(); }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, T> 
    get() const noexcept { return fourth<T>(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, Base> 
    get() const noexcept { return fourth(); }

    // INPUT VALIDATORS
    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_first(T first_) noexcept
    {
        debug_printf(
            "Type reinterpretation during input validation of first field.");
        return is_valid_first(reinterpret_as<Base>(first_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_first(Base first_) noexcept
    {
        return !(first_ & ~((Base{1} << FirstBits) - Base{1}));
    }

    template<class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_second(T second_) noexcept
    {
        debug_printf(
            "Type reinterpretation during input validation of second field.");
        return is_valid_second(reinterpret_as<Base>(second_));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_second(Base second_) noexcept
    {
        return !(second_ & ~((Base{1} << SecondBits) - Base{1}));
    }

    template<
        class T,
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_third(T third_) noexcept
    {
        debug_printf(
            "Type reinterpretation during input validation of third field.");
        return is_valid_third(reinterpret_as<Base>(third_));
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_third(Base third_) noexcept
    {
        return !(third_ & ~((Base{1} << ThirdBits) - Base{1}));
    }

    template<
        class T,
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_fourth(T fourth_) noexcept
    {
        debug_printf(
            "Type reinterpretation during input validation of fourth field.");
        return is_valid_fourth(reinterpret_as<Base>(fourth_));
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_fourth(Base fourth_) noexcept
    {
        return !(fourth_ & ~((Base{1} << FourthBits) - Base{1}));
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 0, bool> 
    is_valid(T first_) noexcept 
    { 
        return is_valid_first(first_); 
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 1, bool> 
    is_valid(T second_) noexcept 
    { 
        return is_valid_second(second_); 
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 2 && ThirdBits, bool> 
    is_valid(T third_) noexcept 
    { 
        return is_valid_third(third_); 
    }

    template<std::size_t I, class T>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, bool> 
    is_valid(T fourth_) noexcept 
    { 
        return is_valid_fourth(fourth_); 
    }

    // OPERATORS
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Pack& operator=(const Pack& pack_) noexcept
    {
        base_ = pack_.base_;
        return *this;
    }
    
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator==(const Pack& pack_) const noexcept
    {
        return base_  == pack_.base_;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr bool operator!=(const Pack& pack_) const noexcept
    {
        return base_ != pack_.base_;
    }

    // CUDA ATOMICS
    // TODO enable_if valid type for CUDA atomics
    DEVICEQUALIFIER INLINEQUALIFIER
    friend Pack atomicCAS(
        Pack * address_, 
        Pack   compare_, 
        Pack   val_) noexcept
    {
        return Pack(atomicCAS(&(address_->base_), compare_.base_, val_.base_));
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    friend Pack atomicExch(
        Pack * address_, 
        Pack   val_) noexcept
    {
        return Pack(atomicExch(&(address_->base_), val_.base_));
    }

private:
    HOSTDEVICEQUALIFIER
    explicit constexpr Pack(Base base) noexcept : base_{base} {}

    Base base_;

}; // class Pack

} // namespace detail

// std::get support
template<std::size_t I, class Base, Base B1, Base B2, Base B3, Base B4>
HOSTDEVICEQUALIFIER INLINEQUALIFIER
constexpr Base get(detail::Pack<Base, B1, B2, B3, B4> pack) noexcept 
{ 
    return pack.template get<I>(); 
}

// packed type aliases
template<class Base, Base FirstBits, Base SecondBits>
using PackedPair = detail::Pack<Base, FirstBits, SecondBits>;

template<class Base, Base FirstBits, Base SecondBits, Base ThirdBits>
using PackedTriple = detail::Pack<Base, FirstBits, SecondBits, ThirdBits>;

template<class Base, Base FirstBits, Base SecondBits, Base ThirdBits, Base FourthBits>
using PackedQuadruple = detail::Pack<Base, FirstBits, SecondBits, ThirdBits, FourthBits>;

#endif /*PACKED_TYPES_CUH*/