#ifndef PACKED_TYPES_CUH
#define PACKED_TYPES_CUH

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

    // bit masks for each individual field
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

    // maximum value for each field to fit into pack
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
    constexpr Pack(Base first_, Base second_) noexcept : base_() 
    {
        first(first_);
        second(second_);
    }

    template<
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
        class   = std::enable_if_t<B1 != 0 && B2 == 0>>
    HOSTDEVICEQUALIFIER
    constexpr Pack(Base first_, Base second_, Base third_) noexcept : base_() 
    {
        first(first_);
        second(second_);
        third(third_);
    }

    template<
        Base B1 = ThirdBits,
        Base B2 = FourthBits,
	    class   = std::enable_if_t<B1 != 0 && B2 != 0>>
    HOSTDEVICEQUALIFIER
    constexpr Pack(Base first_, Base second_, Base third_, Base fourth_) noexcept : base_() 
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
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void first(Base first_) noexcept
    {
        assert(is_valid_first(first_));
        const auto shift = SecondBits + ThirdBits + FourthBits + PaddingBits;
        base_ = (base_ & ~first_mask) + (first_ << shift);
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void second(Base second_) noexcept
    {
        assert(is_valid_second(second_));
        const auto shift = ThirdBits + FourthBits + PaddingBits;
        base_ = (base_ & ~second_mask) + (second_ << shift);
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void third(Base third_) noexcept
    {
        assert(is_valid_third(third_));
        const Base shift = FourthBits + PaddingBits;
        base_ = (base_ & ~third_mask) + (third_ << shift);
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr void fourth(Base fourth_) noexcept
    {
        assert(is_valid_fourth(fourth_));
        const Base shift = PaddingBits;
        base_ = (base_ & ~fourth_mask) + (fourth_ << shift);
    }

    // GETTERS
    // by field name
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base first() const noexcept
    {
        return (base_ >> (SecondBits + ThirdBits + FourthBits + PaddingBits));
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base second() const noexcept
    {
        return ((base_ & second_mask) >> (ThirdBits + FourthBits + PaddingBits));
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base third() const noexcept
    {
        return ((base_ & third_mask) >> (FourthBits + PaddingBits));
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr Base fourth() const noexcept
    {
        return ((base_ & fourth_mask) >> (PaddingBits));
    }

    // SETTERS
    // set<index>(value)
    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, void> 
    set(Base first_) noexcept { first(first_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, void> 
    set(Base second_) noexcept { second(second_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, void> 
    set(Base third_) noexcept { third(third_); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, void> 
    set(Base fourth_) noexcept { fourth(fourth_); }

    // GETTERS
    // get<index>()
    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 0, Base> 
    get() const noexcept { return first(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 1, Base> 
    get() const noexcept { return second(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 2 && ThirdBits, Base> 
    get() const noexcept { return third(); }

    template<std::size_t I>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    constexpr typename std::enable_if_t<I == 3 && ThirdBits && FourthBits, Base> 
    get() const noexcept { return fourth(); }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_first(Base first_) noexcept
    {
        return (first_ <= first_max());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_second(Base second_) noexcept
    {
        return (second_ <= second_max());
    }

    template<
        Base B = ThirdBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_third(Base third_) noexcept
    {
        return (third_ <= third_max());
    }

    template<
        Base B = FourthBits,
	    class  = std::enable_if_t<B != 0>>
    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static constexpr bool is_valid_fourth(Base fourth_) noexcept
    {
        return (fourth_ <= fourth_max());
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool constexpr operator==(const Pack& pack_) const noexcept
    {
        return base_ == pack_.base_;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    bool constexpr operator!=(const Pack& pack_) const noexcept
    {
        return base_ != pack_.base_;
    }

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
Base get(detail::Pack<Base, B1, B2, B3, B4> pack) noexcept 
{ 
    return pack.template get<I>(); 
}

// packed type aliases
template<class Base, Base FirstBits, Base SecondBits>
using PackedPair = detail::Pack<Base, FirstBits, SecondBits>;

template<class Base, Base FirstBits, Base SecondBits, Base ThirdBits, 
    class = std::enable_if_t<ThirdBits>>
using PackedTriple = detail::Pack<Base, FirstBits, SecondBits, ThirdBits>;

template<class Base, Base FirstBits, Base SecondBits, Base ThirdBits, Base FourthBits, 
    class = std::enable_if_t<ThirdBits && FourthBits>>
using PackedQuadruple = detail::Pack<Base, FirstBits, SecondBits, ThirdBits, FourthBits>;

#endif /*PACKED_TYPES_CUH*/