#include "catch.hpp"
#include "packed_types.cuh"

TEMPLATE_TEST_CASE_SIG(
    "PackedPair with variable split", 
    "[pack][pair][packedpair][variablesplit][template]", 
    ((class Base, Base FirstBits, Base SecondBits), 
        Base, FirstBits, SecondBits),
        (std::uint32_t, 16, 16),
        (std::uint32_t, 15, 17),
        (std::uint32_t, 18, 14),
        (std::uint32_t, 7, 7),        
        (std::uint64_t, 32, 32),
        (std::uint64_t, 31, 33),
        (std::uint64_t, 34, 30),
        (std::uint64_t, 7, 7))
{
    using namespace warpcore;

    REQUIRE(FirstBits + SecondBits <= sizeof(Base) * 8);

    using pack_t = PackedPair<Base, FirstBits, SecondBits>;
    using base_t   = typename pack_t::base_type;
    
    const base_t first_max = pack_t::first_max();
    const base_t second_max = pack_t::second_max();
    const base_t first  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::first_max());
    const base_t second  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::second_max());
    const base_t update = 60;

    CAPTURE(first, second, update, first_max, second_max);

    REQUIRE(first  <= first_max);
    REQUIRE(second <= second_max);
    REQUIRE(update <= first_max);
    REQUIRE(update <= second_max);

    CHECK(pack_t::is_valid_first(first));
    CHECK(pack_t::is_valid_second(second));
    CHECK(pack_t::is_valid_first(update));
    CHECK(pack_t::is_valid_second(update));

    SECTION("pack size")
    {
        CHECK(sizeof(pack_t) == sizeof(base_t));
    }

    SECTION("empty pack")
    {
        pack_t empty     = pack_t();
        pack_t empty_too = pack_t::empty();

        CHECK(empty == empty_too);
        CHECK(empty.get_first() == 0);
        CHECK(empty.get_second() == 0);
    }

    SECTION("set and get pack")
    {
        pack_t pack(first, second);

        CHECK(pack.get_first() == first);
        CHECK(pack.get_second() == second);

        SECTION("equality operator")
        {
            pack_t pack_too = pack;

            CHECK(pack == pack_too);
        }

        SECTION("update first")
        {
            pack.set_first(update);

            CHECK(pack.get_first() == update);
            CHECK(pack.get_second() == second);
        }

        SECTION("update second")
        {
            pack.set_second(update);

            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == update);
        }

        SECTION("maximum first value")
        {
            pack.set_first(first_max);

            CHECK(first_max != 0);
            CHECK(pack.get_first() == first_max);
            CHECK(pack.get_second() == second);
        }

        SECTION("maximum second value")
        {
            pack.set_second(second_max);

            CHECK(second_max != 0);
            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second_max);
        }

        SECTION("atomic operations")
        {
            pack_t val = pack_t(update, update);

            pack_t * pack_d = nullptr;
            cudaMalloc(&pack_d, sizeof(pack_t));
            REQUIRE(cudaGetLastError() == cudaSuccess);

            SECTION("atomic CAS")
            {
                pack_t compare = pack_t(pack);

                cudaMemcpy(pack_d, &pack, sizeof(pack_t), H2D);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                lambda_kernel
                <<<1, 1>>>([=] DEVICEQUALIFIER
                {
                    atomicCAS(pack_d, compare, val);
                });

                cudaMemcpy(&pack, pack_d, sizeof(pack_t), D2H);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                CHECK(pack == val);
            }

            SECTION("atomic exchange")
            {
                cudaMemcpy(pack_d, &pack, sizeof(pack_t), H2D);
                REQUIRE(cudaGetLastError() == cudaSuccess);
    
                lambda_kernel
                <<<1, 1>>>([=] DEVICEQUALIFIER
                {
                    atomicExch(pack_d, val);
                });

                cudaMemcpy(&pack, pack_d, sizeof(pack_t), D2H);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                CHECK(pack == val);
            }

            cudaFree(pack_d);
            CHECK(cudaGetLastError() == cudaSuccess);
        }
    }
}

TEMPLATE_TEST_CASE_SIG(
    "PackedTriple with variable split", 
    "[pack][triple][packedtriple][variablesplit][template]", 
    ((class Base, Base FirstBits, Base SecondBits, Base ThirdBits), 
        Base, FirstBits, SecondBits, ThirdBits), 
        (std::uint32_t, 10, 10, 12),
        (std::uint32_t, 8, 9, 15),
        (std::uint32_t, 13, 8, 11),
        (std::uint32_t, 7, 7, 7),
        (std::uint64_t, 20, 20, 24),
        (std::uint64_t, 18, 19, 27),
        (std::uint64_t, 23, 19, 22),
        (std::uint64_t, 7, 7, 7))
{
    using namespace warpcore;

    REQUIRE(FirstBits + SecondBits + ThirdBits <= sizeof(Base) * 8);

    using pack_t = PackedTriple<Base, FirstBits, SecondBits, ThirdBits>;
    using base_t = typename pack_t::base_type;
    
    const base_t first_max = pack_t::first_max();
    const base_t second_max = pack_t::second_max();
    const base_t third_max = pack_t::third_max();
    const base_t first  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::first_max());
    const base_t second  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::second_max());
    const base_t third  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::third_max());
    const base_t update = 60;

    CAPTURE(first, second, third, update, first_max, second_max, third_max);

    REQUIRE(first  <= first_max);
    REQUIRE(second <= second_max);
    REQUIRE(third  <= third_max);
    REQUIRE(update <= first_max);
    REQUIRE(update <= second_max);
    REQUIRE(update <= third_max);

    CHECK(pack_t::is_valid_first(first));
    CHECK(pack_t::is_valid_second(second));
    CHECK(pack_t::is_valid_third(third));
    CHECK(pack_t::is_valid_first(update));
    CHECK(pack_t::is_valid_second(update));
    CHECK(pack_t::is_valid_third(update));

    SECTION("pack size")
    {
        CHECK(sizeof(pack_t) == sizeof(base_t));
    }

    SECTION("empty pack")
    {
        pack_t empty     = pack_t();
        pack_t empty_too = pack_t::empty();

        CHECK(empty == empty_too);
        CHECK(empty.get_first() == 0);
        CHECK(empty.get_second() == 0);
        CHECK(empty.get_third() == 0);
    }

    SECTION("set and get pack")
    {
        pack_t pack(first, second, third);

        CHECK(pack.get_first() == first);
        CHECK(pack.get_second() == second);
        CHECK(pack.get_third() == third);

        SECTION("equality operator")
        {
            pack_t pack_too = pack;

            CHECK(pack == pack_too);
        }

        SECTION("update first")
        {
            pack.set_first(update);

            CHECK(pack.get_first() == update);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third);
        }

        SECTION("update second")
        {
            pack.set_second(update);

            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == update);
            CHECK(pack.get_third() == third);
        }

        SECTION("update third")
        {
            pack.set_third(update);

            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == update);
        }

        SECTION("maximum first value")
        {
            pack.set_first(first_max);

            CHECK(first_max != 0);
            CHECK(pack.get_first() == first_max);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third);
        }

        SECTION("maximum second value")
        {
            pack.set_second(second_max);

            CHECK(second_max != 0);
            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second_max);
            CHECK(pack.get_third() == third);
        }

        SECTION("maximum third value")
        {
            pack.set_third(third_max);

            CHECK(third_max != 0);
            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third_max);
        }

        SECTION("atomic operations")
        {
            pack_t val = pack_t(update, update, update);

            pack_t * pack_d = nullptr;
            cudaMalloc(&pack_d, sizeof(pack_t));
            REQUIRE(cudaGetLastError() == cudaSuccess);

            SECTION("atomic CAS")
            {
                pack_t compare = pack_t(pack);

                cudaMemcpy(pack_d, &pack, sizeof(pack_t), H2D);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                lambda_kernel
                <<<1, 1>>>([=] DEVICEQUALIFIER
                {
                    atomicCAS(pack_d, compare, val);
                });

                cudaMemcpy(&pack, pack_d, sizeof(pack_t), D2H);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                CHECK(pack == val);
            }

            SECTION("atomic exchange")
            {
                cudaMemcpy(pack_d, &pack, sizeof(pack_t), H2D);
                REQUIRE(cudaGetLastError() == cudaSuccess);
    
                lambda_kernel
                <<<1, 1>>>([=] DEVICEQUALIFIER
                {
                    atomicExch(pack_d, val);
                });

                cudaMemcpy(&pack, pack_d, sizeof(pack_t), D2H);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                CHECK(pack == val);
            }

            cudaFree(pack_d);
            CHECK(cudaGetLastError() == cudaSuccess);
        }
    }
}

TEMPLATE_TEST_CASE_SIG(
    "PackedQuadruple with variable split", 
    "[pack][quadruple][packedquadruple][variablesplit][template]", 
    ((class Base, Base FirstBits, Base SecondBits, Base ThirdBits, Base FourthBits), 
        Base, FirstBits, SecondBits, ThirdBits, FourthBits), 
        (std::uint32_t, 8, 8, 8, 8),
        (std::uint32_t, 7, 9, 9, 7),
        (std::uint32_t, 9, 8, 7, 8),
        (std::uint32_t, 7, 7, 7, 7),
        (std::uint64_t, 16, 16, 16, 16),
        (std::uint64_t, 15, 17, 13, 19),
        (std::uint64_t, 8, 8, 32, 16),
        (std::uint64_t, 7, 7, 7, 7))
{
    using namespace warpcore;

    REQUIRE(FirstBits + SecondBits + ThirdBits + FourthBits <= sizeof(Base) * 8);

    using pack_t = PackedQuadruple<Base, FirstBits, SecondBits, ThirdBits, FourthBits>;
    using base_t = typename pack_t::base_type;
    
    const base_t first_max = pack_t::first_max();
    const base_t second_max = pack_t::second_max();
    const base_t third_max = pack_t::third_max();
    const base_t fourth_max = pack_t::fourth_max();
    const base_t first  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::first_max());
    const base_t second  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::second_max());
    const base_t third  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::third_max());
    const base_t fourth  = GENERATE(as<base_t>{}, 0, 1, 2, 42, pack_t::fourth_max());
    const base_t update = 60;

    CAPTURE(first, second, third, fourth, update, first_max, second_max, third_max, fourth_max);

    REQUIRE(first   <= first_max);
    REQUIRE(second  <= second_max);
    REQUIRE(third   <= third_max);
    REQUIRE(fourth  <= fourth_max);
    REQUIRE(update  <= first_max);
    REQUIRE(update  <= second_max);
    REQUIRE(update  <= third_max);
    REQUIRE(update  <= fourth_max);

    CHECK(pack_t::is_valid_first(first));
    CHECK(pack_t::is_valid_second(second));
    CHECK(pack_t::is_valid_third(third));
    CHECK(pack_t::is_valid_fourth(fourth));
    CHECK(pack_t::is_valid_first(update));
    CHECK(pack_t::is_valid_second(update));
    CHECK(pack_t::is_valid_third(update));
    CHECK(pack_t::is_valid_fourth(update));

    SECTION("pack size")
    {
        CHECK(sizeof(pack_t) == sizeof(base_t));
    }

    SECTION("empty pack")
    {
        pack_t empty     = pack_t();
        pack_t empty_too = pack_t::empty();

        CHECK(empty == empty_too);
        CHECK(empty.get_first() == 0);
        CHECK(empty.get_second() == 0);
        CHECK(empty.get_third() == 0);
        CHECK(empty.get_fourth() == 0);
    }

    SECTION("set and get pack")
    {
        pack_t pack(first, second, third, fourth);

        CHECK(pack.get_first() == first);
        CHECK(pack.get_second() == second);
        CHECK(pack.get_third() == third);
        CHECK(pack.get_fourth() == fourth);

        SECTION("equality operator")
        {
            pack_t pack_too = pack;

            CHECK(pack == pack_too);
        }

        SECTION("update first")
        {
            pack.set_first(update);

            CHECK(pack.get_first() == update);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third);
            CHECK(pack.get_fourth() == fourth);
        }

        SECTION("update second")
        {
            pack.set_second(update);

            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == update);
            CHECK(pack.get_third() == third);
            CHECK(pack.get_fourth() == fourth);
        }

        SECTION("update third")
        {
            pack.set_third(update);

            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == update);
            CHECK(pack.get_fourth() == fourth);
        }

        SECTION("update fourth")
        {
            pack.set_fourth(update);

            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third);
            CHECK(pack.get_fourth() == update);
        }

        SECTION("maximum first value")
        {
            pack.set_first(first_max);

            CHECK(first_max != 0);
            CHECK(pack.get_first() == first_max);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third);
            CHECK(pack.get_fourth() == fourth);
        }

        SECTION("maximum second value")
        {
            pack.set_second(second_max);

            CHECK(second_max != 0);
            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second_max);
            CHECK(pack.get_third() == third);
            CHECK(pack.get_fourth() == fourth);
        }

        SECTION("maximum third value")
        {
            pack.set_third(third_max);

            CHECK(third_max != 0);
            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third_max);
            CHECK(pack.get_fourth() == fourth);
        }

        SECTION("maximum third value")
        {
            pack.set_fourth(fourth_max);

            CHECK(third_max != 0);
            CHECK(pack.get_first() == first);
            CHECK(pack.get_second() == second);
            CHECK(pack.get_third() == third);
            CHECK(pack.get_fourth() == fourth_max);
        }

        SECTION("atomic operations")
        {
            pack_t val = pack_t(update, update, update, update);

            pack_t * pack_d = nullptr;
            cudaMalloc(&pack_d, sizeof(pack_t));
            REQUIRE(cudaGetLastError() == cudaSuccess);

            SECTION("atomic CAS")
            {
                pack_t compare = pack_t(pack);

                cudaMemcpy(pack_d, &pack, sizeof(pack_t), H2D);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                lambda_kernel
                <<<1, 1>>>([=] DEVICEQUALIFIER
                {
                    atomicCAS(pack_d, compare, val);
                });

                cudaMemcpy(&pack, pack_d, sizeof(pack_t), D2H);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                CHECK(pack == val);
            }

            SECTION("atomic exchange")
            {
                cudaMemcpy(pack_d, &pack, sizeof(pack_t), H2D);
                REQUIRE(cudaGetLastError() == cudaSuccess);
    
                lambda_kernel
                <<<1, 1>>>([=] DEVICEQUALIFIER
                {
                    atomicExch(pack_d, val);
                });

                cudaMemcpy(&pack, pack_d, sizeof(pack_t), D2H);
                REQUIRE(cudaGetLastError() == cudaSuccess);

                CHECK(pack == val);
            }

            cudaFree(pack_d);
            CHECK(cudaGetLastError() == cudaSuccess);
        }
    }
}