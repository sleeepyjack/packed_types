#pragma once

#include <cstdint>
#include <type_traits>
#include <assert.h>
#include "cudahelpers/cuda_helpers.cuh"

// INFO you can find the actual types as using statements at the end of this file

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
    // memory layout: <first|second|third|fourth|padding>

    static_assert(FirstBits != 0 && SecondBits != 0,
        "FirstBits and SecondBits both may not be zero.");

    static_assert(!(ThirdBits == 0 && FourthBits != 0),
        "Third type cannot be zero-width if fourth type has non-zero width.");

    static_assert(FirstBits + SecondBits + ThirdBits + FourthBits <= sizeof(Base) * 8,
        "Too many bits for chosen datatype.");

    static_assert(std::is_fundamental<Base>::value,
        "Base must be fundamental type.");

    static_assert(std::is_unsigned<Base>::value,
        "Base must be unsigned type.");


    static constexpr Base PaddingBits = 
        (sizeof(Base) * 8) - (FirstBits + SecondBits + ThirdBits + FourthBits);


    static constexpr Base first_mask = 
        ((Base{1} << FirstBits) - 1) << (SecondBits + ThirdBits + FourthBits + PaddingBits);

    static constexpr Base second_mask = 
        ((Base{1} << SecondBits) - 1) << (ThirdBits + FourthBits + PaddingBits);

    static constexpr Base third_mask = 
        (ThirdBits == 0) ? 
            Base{0} : 
            ((Base{1} << ThirdBits) - 1) << (FourthBits + PaddingBits);

    static constexpr Base fourth_mask = 
        (FourthBits == 0) ? 
            Base{0} : 
            ((Base{1} << FourthBits) - 1) << (PaddingBits);

    static constexpr Base padding_mask = 
        (PaddingBits == 0) ? 
            Base{0} : 
            ((Base{1} << PaddingBits) - 1);


public:
    using base_type = Base;

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


    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base first_max() noexcept
    { 
        return (Base{1} << FirstBits) - 1; 
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base second_max() noexcept 
    { 
        return (Base{1} << SecondBits) - 1; 
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base third_max() noexcept 
    { 
        return (Base{1} << ThirdBits) - 1; 
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Base fourth_max() noexcept 
    { 
        return (Base{1} << FourthBits) - 1; 
    }

    HOSTDEVICEQUALIFIER
    constexpr Pack() noexcept : base_{empty().base_} {}

    template<
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
	    class   = std::enable_if_t<B1 == 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER
    constexpr Pack(Base first, Base second) noexcept : base_() 
    {
        set_first(first);
        set_second(second);
    }

    template<
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
        class   = std::enable_if_t<B1 != 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER
    constexpr Pack(Base first, Base second, Base third) noexcept : base_() 
    {
        set_first(first);
        set_second(second);
        set_third(third);
    }

    template<
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
	    class   = std::enable_if_t<B1 != 0 && B2 != 0>>
    HOSTDEVICEQUALIFIER
    constexpr Pack(Base first, Base second, Base third, Base fourth) noexcept : base_() 
    {
        set_first(first);
        set_second(second);
        set_third(third);
        set_fourth(fourth);
    }

    constexpr Pack(const Pack&) noexcept = default;
    constexpr Pack(Pack&& pair) noexcept = default;

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr Pack empty() noexcept 
    { 
        return Pack(Base{0});
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void set_first(Base first) noexcept
    {
        assert(is_valid_first(first));
        const auto shift = SecondBits + ThirdBits + FourthBits + PaddingBits;
        base_ = (base_ & ~first_mask) + (first << shift);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void set_second(Base second) noexcept
    {
        assert(is_valid_second(second));
        const auto shift = ThirdBits + FourthBits + PaddingBits;
        base_ = (base_ & ~second_mask) + (second << shift);
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void set_third(Base third) noexcept
    {
        assert(is_valid_third(third));
        const Base shift = FourthBits + PaddingBits;
        base_ = (base_ & ~third_mask) + (third << shift);
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void set_fourth(Base fourth) noexcept
    {
        assert(is_valid_fourth(fourth));
        const Base shift = PaddingBits;
        base_ = (base_ & ~fourth_mask) + (fourth << shift);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base get_first() const noexcept
    {
        return (base_ >> (SecondBits + ThirdBits + FourthBits + PaddingBits));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base get_second() const noexcept
    {
        return ((base_ & second_mask) >> (ThirdBits + FourthBits + PaddingBits));
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base get_third() const noexcept
    {
        return ((base_ & third_mask) >> (FourthBits + PaddingBits));
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base get_fourth() const noexcept
    {
        return ((base_ & fourth_mask) >> (PaddingBits));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_first(Base first) noexcept
    {
        return (first <= first_max());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_second(Base second) noexcept
    {
        return (second <= second_max());
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_third(Base third) noexcept
    {
        return (third <= third_max());
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_fourth(Base fourth) noexcept
    {
        return (fourth <= fourth_max());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool constexpr operator==(const Pack& base) const noexcept
    {
        return base_ == base.base_;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool constexpr operator!=(const Pack& base) const noexcept
    {
        return base_ != base.base_;
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    friend Pack atomicCAS(
        Pack * address, 
        Pack   compare, 
        Pack   val) noexcept
    {
        return Pack(atomicCAS(&(address->base_), compare.base_, val.base_));
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    friend Pack atomicExch(
        Pack * address, 
        Pack   val) noexcept
    {
        return Pack(atomicExch(&(address->base_), val.base_));
    }

private:
    HOSTDEVICEQUALIFIER
    explicit constexpr Pack(Base base) noexcept : base_{base} {}

    Base base_;

}; // class Pack

} // namespace detail

template<class Base, Base FirstBits, Base SecondBits>
using PackedPair = detail::Pack<Base, FirstBits, SecondBits>;

template<class Base, Base FirstBits, Base SecondBits, Base ThirdBits>
using PackedTriple = detail::Pack<Base, FirstBits, SecondBits, ThirdBits>;

template<class Base, Base FirstBits, Base SecondBits, Base ThirdBits, Base FourthBits>
using PackedQuadruple = detail::Pack<Base, FirstBits, SecondBits, ThirdBits, FourthBits>;