// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file impl_variants.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Runtime selectable implementations of sections of the SPH solver.
 */

#include "shambase/exception.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/impl_variants.hpp"
#include "shamsys/NodeInstance.hpp"
#include <map>

namespace shammodels::sph::impl {

    namespace {

        shamalgs::ImplVariantGlobal<diff_operators::SeparateKernels, diff_operators::FusedKernel>
            diff_operators_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(diff_operators::FusedKernel{});
            }};

        shamalgs::ImplVariantGlobal<cfl_vsig::SeparatePass, cfl_vsig::FusedWithDerivs>
            cfl_vsig_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(cfl_vsig::FusedWithDerivs{});
            }};

        shamalgs::ImplVariantGlobal<
            neigh_cache_particle_pass::AllLeafParticles,
            neigh_cache_particle_pass::PruneLeaves>
            neigh_cache_particle_pass_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(neigh_cache_particle_pass::PruneLeaves{});
            }};

        shamalgs::ImplVariantGlobal<
            neigh_cache_particle_layout::Compact,
            neigh_cache_particle_layout::Slots>
            neigh_cache_particle_layout_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(neigh_cache_particle_layout::Slots{});
            }};

        shamalgs::ImplVariantGlobal<
            neigh_cache_candidate_data::Indirect,
            neigh_cache_candidate_data::LeafSortedCopy>
            neigh_cache_candidate_data_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(neigh_cache_candidate_data::LeafSortedCopy{});
            }};

        shamalgs::ImplVariantGlobal<
            neigh_cache_leaf_pass::CountThenFill,
            neigh_cache_leaf_pass::SingleTraversal>
            neigh_cache_leaf_pass_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                self.set(neigh_cache_leaf_pass::SingleTraversal{});
            }};

        shamalgs::
            ImplVariantGlobal<neigh_loop_arithmetic::Divisions, neigh_loop_arithmetic::Reciprocals>
                neigh_loop_arithmetic_impl{[](const sham::DeviceScheduler_ptr &, auto &self) {
                    self.set(neigh_loop_arithmetic::Reciprocals{});
                }};

        /// Registry of the selectors, by section name
        const std::map<std::string, shamalgs::IImplVariant *> &get_registry() {
            static const std::map<std::string, shamalgs::IImplVariant *> registry{
                {"diff_operators", &diff_operators_impl},
                {"cfl_vsig", &cfl_vsig_impl},
                {"neigh_cache_particle_pass", &neigh_cache_particle_pass_impl},
                {"neigh_cache_particle_layout", &neigh_cache_particle_layout_impl},
                {"neigh_cache_candidate_data", &neigh_cache_candidate_data_impl},
                {"neigh_cache_leaf_pass", &neigh_cache_leaf_pass_impl},
                {"neigh_loop_arithmetic", &neigh_loop_arithmetic_impl},
            };
            return registry;
        }

        shamalgs::IImplVariant &get_section(const std::string &section) {
            auto &registry = get_registry();
            auto it        = registry.find(section);
            if (it == registry.end()) {
                throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                    "unknown SPH implementation section : {}, possible sections : {}",
                    section,
                    get_impl_sections()));
            }
            return *it->second;
        }

        template<class Selector>
        const typename Selector::Variant &get_or_autoselect(Selector &sel) {
            if (!sel.is_set()) {
                sel.autoselect(shamsys::instance::get_compute_scheduler_ptr());
            }
            return sel.get();
        }

    } // namespace

    const diff_operators::Variant &get_impl_diff_operators() {
        return get_or_autoselect(diff_operators_impl);
    }

    const cfl_vsig::Variant &get_impl_cfl_vsig() { return get_or_autoselect(cfl_vsig_impl); }

    const neigh_cache_particle_pass::Variant &get_impl_neigh_cache_particle_pass() {
        return get_or_autoselect(neigh_cache_particle_pass_impl);
    }

    const neigh_cache_particle_layout::Variant &get_impl_neigh_cache_particle_layout() {
        return get_or_autoselect(neigh_cache_particle_layout_impl);
    }

    const neigh_cache_candidate_data::Variant &get_impl_neigh_cache_candidate_data() {
        return get_or_autoselect(neigh_cache_candidate_data_impl);
    }

    const neigh_cache_leaf_pass::Variant &get_impl_neigh_cache_leaf_pass() {
        return get_or_autoselect(neigh_cache_leaf_pass_impl);
    }

    const neigh_loop_arithmetic::Variant &get_impl_neigh_loop_arithmetic() {
        return get_or_autoselect(neigh_loop_arithmetic_impl);
    }

    std::vector<std::string> get_impl_sections() {
        std::vector<std::string> ret;
        for (auto &[name, sel] : get_registry()) {
            ret.push_back(name);
        }
        return ret;
    }

    std::vector<std::string> get_default_impl_list(const std::string &section) {
        return get_section(section).get_default_config_list();
    }

    std::string get_current_impl(const std::string &section) {
        return get_section(section).get_current_config();
    }

    void set_impl(const std::string &section, const std::string &impl) {
        shamlog_info_ln("SPH", "setting", section, "implementation to impl :", impl);
        get_section(section).set(impl);
    }

    void autoselect_all_impl() {
        for (auto &[name, sel] : get_registry()) {
            sel->autoselect(shamsys::instance::get_compute_scheduler_ptr());
        }
    }

} // namespace shammodels::sph::impl
