#include "CL/sycl/range.hpp"
#include "aliases.hpp"
#include "interfaces/interface_generator.hpp"
#include "interfaces/interface_handler.hpp"
#include "interfaces/interface_selector.hpp"
#include "io/dump.hpp"
#include "patch/patch_field.hpp"
#include "patch/patch_reduc_tree.hpp"
#include "patch/patchdata.hpp"
#include "patch/serialpatchtree.hpp"
#include "patch/patchdata_exchanger.hpp"
#include "patchscheduler/loadbalancing_hilbert.hpp"
#include "patchscheduler/patch_content_exchanger.hpp"
#include "patchscheduler/scheduler_mpi.hpp"
#include "sys/cmdopt.hpp"
#include "sys/mpi_handler.hpp"
#include "sys/sycl_mpi_interop.hpp"
#include "unittests/shamrocktest.hpp"
#include "utils/string_utils.hpp"
#include <memory>
#include <mpi.h>
#include <string>
#include <unordered_map>
#include <vector>


int main(int argc, char *argv[]){


    std::cout << shamrock_title_bar_big << std::endl;

    mpi_handler::init();

    Cmdopt & opt = Cmdopt::get_instance();
    opt.init(argc, argv,"./shamrock");

    SyCLHandler & hndl = SyCLHandler::get_instance();
    hndl.init_sycl();

    

    SchedulerMPI sched = SchedulerMPI(2000,1);
    sched.init_mpi_required_types();

    patchdata_layout::set(1, 0, 0, 0, 0, 0);
    patchdata_layout::sync(MPI_COMM_WORLD);

    if (mpi_handler::world_rank == 0) {
        Patch p;

        p.data_count    = 1e5;
        p.load_value    = 1e5;
        p.node_owner_id = mpi_handler::world_rank;

        p.x_min = 0;
        p.y_min = 0;
        p.z_min = 0;

        p.x_max = HilbertLB::max_box_sz;
        p.y_max = HilbertLB::max_box_sz;
        p.z_max = HilbertLB::max_box_sz;

        p.pack_node_index = u64_max;

        PatchData pdat;

        std::mt19937 eng(0x1111);
        std::uniform_real_distribution<f32> distpos(-1, 1);

        for (u32 part_id = 0; part_id < p.data_count; part_id++)
            pdat.pos_s.push_back({distpos(eng), distpos(eng), distpos(eng)});

        sched.add_patch(p, pdat);

    } else {
        sched.patch_list._next_patch_id++;
    }
    
    sched.owned_patch_id = sched.patch_list.build_local();

    // std::cout << sched.dump_status() << std::endl;
    sched.patch_list.build_global();
    // std::cout << sched.dump_status() << std::endl;

    //*
    sched.patch_tree.build_from_patchtable(sched.patch_list.global, HilbertLB::max_box_sz);
    sched.patch_data.sim_box.min_box_sim_s = {-1};
    sched.patch_data.sim_box.max_box_sim_s = {1};

    // std::cout << sched.dump_status() << std::endl;

    std::cout << "build local" << std::endl;
    sched.owned_patch_id = sched.patch_list.build_local();
    sched.patch_list.build_local_idx_map();
    sched.update_local_dtcnt_value();
    sched.update_local_load_value();

    // sched.patch_list.build_global();

    std::cout << " ------ step time = " << 0 << " ------" << std::endl;
    {
        SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());
        sptree.attach_buf();

        PatchField<f32> h_field;
        h_field.local_nodes_value.resize(sched.patch_list.local.size());
        for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
            h_field.local_nodes_value[idx] = 0.1f;
        }
        h_field.build_global(mpi_type_f32);

        InterfaceHandler<f32_3, f32> interface_hndl;
        interface_hndl.compute_interface_list<InterfaceSelector_SPH<f32_3, f32>>(sched, sptree, h_field);
        interface_hndl.comm_interfaces(sched);
        interface_hndl.print_current_interf_map();

        //sched.dump_local_patches(format("patches_%d_node%d", 0, mpi_handler::world_rank));

        dump_state("step"+std::to_string(0)+"/",sched);
    }

    for (u32 stepi = 1; stepi < 6; stepi++) {
        std::cout << " ------ step time = " << stepi << " ------" << std::endl;
        // std::cout << sched.dump_status() << std::endl;
        sched.scheduler_step(true, true);

        std::cout << " reduc " << std::endl;
        {

            // std::cout << sched.dump_status() << std::endl;

            // PatchField<u64> dtcnt_field;
            // dtcnt_field.local_nodes_value.resize(sched.patch_list.local.size());
            // for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
            //     dtcnt_field.local_nodes_value[idx] = sched.patch_list.local[idx].data_count;
            // }

            // std::cout << "dtcnt_field.build_global(mpi_type_u64);" << std::endl;
            // dtcnt_field.build_global(mpi_type_u64);

            // std::cout << "len 1 : " << dtcnt_field.local_nodes_value.size() << std::endl;
            // std::cout << "len 2 : " << dtcnt_field.global_values.size() << std::endl;

            SerialPatchTree<f32_3> sptree(sched.patch_tree, sched.get_box_tranform<f32_3>());
            // sptree.dump_dat();

            // std::cout << "len 3 : " << sptree.get_element_count() << std::endl;

            // std::cout << "sptree.attach_buf();" << std::endl;
            sptree.attach_buf();

            // std::cout << "sptree.reduce_field" << std::endl;
            // PatchFieldReduction<u64> pfield_reduced =
            //     sptree.reduce_field<u64, Reduce_DataCount>(hndl.get_queue_alt(0), sched, dtcnt_field);

            // std::cout << "pfield_reduced.detach_buf()" << std::endl;
            // pfield_reduced.detach_buf();
            // std::cout << " ------ > " << pfield_reduced.tree_field[0] << "\n\n\n";

            PatchField<f32> h_field;
            h_field.local_nodes_value.resize(sched.patch_list.local.size());
            for (u64 idx = 0; idx < sched.patch_list.local.size(); idx++) {
                h_field.local_nodes_value[idx] = 0.1f;
            }
            h_field.build_global(mpi_type_f32);

            InterfaceHandler<f32_3, f32> interface_hndl;
            interface_hndl.compute_interface_list<InterfaceSelector_SPH<f32_3, f32>>(sched, sptree, h_field);
            interface_hndl.comm_interfaces(sched);
            //interface_hndl.print_current_interf_map();

            //sched.dump_local_patches(format("patches_%d_node%d", stepi, mpi_handler::world_rank));

            for(auto & [id,pdat] : sched.patch_data.owned_data){
                if(pdat.pos_s.size() > 0){
                    std::unique_ptr<sycl::buffer<f32_3>> pos = std::make_unique<sycl::buffer<f32_3>>(pdat.pos_s.data(),pdat.pos_s.size());


                    if(true && stepi > 2){
                        hndl.get_queue_compute(0).submit([&](sycl::handler &cgh) {
                            auto posacc= pos->get_access<sycl::access::mode::read_write>(cgh);
                            
                            cgh.parallel_for<class Modify_pos>(sycl::range(pos->size()), [=](sycl::item<1> item) {
                                u32 i = (u32)item.get_id(0);

                                posacc[i] *= 1.1; 
                            });
                        });
                    }
                }
            }


            

            bool err_id_in_newid = false;
            std::unordered_map<u64, sycl::buffer<u64>> newid_buf_map;
            for(auto & [id,pdat] : sched.patch_data.owned_data ){
                std::unique_ptr<sycl::buffer<f32_3>> pos = std::make_unique<sycl::buffer<f32_3>>(pdat.pos_s.data(),pdat.pos_s.size());

                newid_buf_map.insert({
                    id,
                    __compute_object_patch_owner<f32_3, class ComputeObejctPatchOwners>(
                        hndl.get_queue_compute(0), 
                        *pos, 
                        sptree)});

                pos.reset();

                
                {
                    auto nid = newid_buf_map.at(id).get_access<sycl::access::mode::read>();
                    for(u32 i = 0 ; i < pdat.pos_s.size() ; i++){
                        err_id_in_newid = err_id_in_newid || (nid[i] == u64_max);
                    }
                }
                
            }

            printf("err_id_in_newid : %d \n", err_id_in_newid);

            if(sched.should_resize_box(err_id_in_newid)){
                sched.patch_data.sim_box.reset_box_size();
                
                if(patchdata_layout::nVarpos_s == 1){
                    for(auto & [id,pdat] : sched.patch_data.owned_data ){
                        for(f32_3 & r : pdat.pos_s){
                            sched.patch_data.sim_box.min_box_sim_s = sycl::min(sched.patch_data.sim_box.min_box_sim_s,r);
                            sched.patch_data.sim_box.max_box_sim_s = sycl::max(sched.patch_data.sim_box.max_box_sim_s,r);
                        }
                    }
                    f32_3 new_minbox = sched.patch_data.sim_box.min_box_sim_s;
                    f32_3 new_maxbox = sched.patch_data.sim_box.max_box_sim_s;
                    mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_s.x(), &new_minbox.x(), 1 , mpi_type_f32,MPI_MIN, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_s.y(), &new_minbox.y(), 1 , mpi_type_f32,MPI_MIN, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_s.z(), &new_minbox.z(), 1 , mpi_type_f32,MPI_MIN, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_s.x(), &new_maxbox.x(), 1 , mpi_type_f32,MPI_MAX, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_s.y(), &new_maxbox.y(), 1 , mpi_type_f32,MPI_MAX, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_s.z(), &new_maxbox.z(), 1 , mpi_type_f32,MPI_MAX, MPI_COMM_WORLD);

                    sched.patch_data.sim_box.min_box_sim_s = new_minbox;
                    sched.patch_data.sim_box.max_box_sim_s = new_maxbox;

                    printf("resize box to  : {%f,%f,%f,%f,%f,%f}\n",new_minbox.x(),new_minbox.y(),new_minbox.z(),new_maxbox.x(),new_maxbox.y(),new_maxbox.z());
                    sched.patch_data.sim_box.clean_box<f32>(1.2);

                    new_minbox = sched.patch_data.sim_box.min_box_sim_s;
                    new_maxbox = sched.patch_data.sim_box.max_box_sim_s;
                    printf("resize box to  : {%f,%f,%f,%f,%f,%f}\n",new_minbox.x(),new_minbox.y(),new_minbox.z(),new_maxbox.x(),new_maxbox.y(),new_maxbox.z());

                }

                if(patchdata_layout::nVarpos_d == 1){
                    for(auto & [id,pdat] : sched.patch_data.owned_data ){
                        for(f64_3 & r : pdat.pos_d){
                            sched.patch_data.sim_box.min_box_sim_d = sycl::min(sched.patch_data.sim_box.min_box_sim_d,r);
                            sched.patch_data.sim_box.max_box_sim_d = sycl::max(sched.patch_data.sim_box.max_box_sim_d,r);
                        }
                    }

                    f64_3 new_minbox = sched.patch_data.sim_box.min_box_sim_d;
                    f64_3 new_maxbox = sched.patch_data.sim_box.max_box_sim_d;

                    mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_d.x(), &new_minbox.x(), 1 , mpi_type_f64,MPI_MIN, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_d.y(), &new_minbox.y(), 1 , mpi_type_f64,MPI_MIN, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.min_box_sim_d.z(), &new_minbox.z(), 1 , mpi_type_f64,MPI_MIN, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_d.x(), &new_maxbox.x(), 1 , mpi_type_f64,MPI_MAX, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_d.y(), &new_maxbox.y(), 1 , mpi_type_f64,MPI_MAX, MPI_COMM_WORLD);
                    mpi::allreduce(&sched.patch_data.sim_box.max_box_sim_d.z(), &new_maxbox.z(), 1 , mpi_type_f64,MPI_MAX, MPI_COMM_WORLD);

                    sched.patch_data.sim_box.min_box_sim_d = new_minbox;
                    sched.patch_data.sim_box.max_box_sim_d = new_maxbox;

                    printf("resize box to  : {%f,%f,%f,%f,%f,%f}\n",new_minbox.x(),new_minbox.y(),new_minbox.z(),new_maxbox.x(),new_maxbox.y(),new_maxbox.z());
                    sched.patch_data.sim_box.clean_box<f64>(1.2);

                    new_minbox = sched.patch_data.sim_box.min_box_sim_d;
                    new_maxbox = sched.patch_data.sim_box.max_box_sim_d;
                    printf("resize box to  : {%f,%f,%f,%f,%f,%f}\n",new_minbox.x(),new_minbox.y(),new_minbox.z(),new_maxbox.x(),new_maxbox.y(),new_maxbox.z());

                }

                sptree.detach_buf();
                sptree = SerialPatchTree<f32_3>(sched.patch_tree, sched.get_box_tranform<f32_3>());
                sptree.attach_buf();

                for(auto & [id,pdat] : sched.patch_data.owned_data ){
                    std::unique_ptr<sycl::buffer<f32_3>> pos = std::make_unique<sycl::buffer<f32_3>>(pdat.pos_s.data(),pdat.pos_s.size());

                    newid_buf_map.at(id)=
                        __compute_object_patch_owner<f32_3, class ComputeObejctPatchOwners2>(
                            hndl.get_queue_compute(0), 
                            *pos, 
                            sptree);

                    pos.reset();
                    
                }
                
            }

           



            std::vector<std::unique_ptr<PatchData>> comm_pdat;
            std::vector<u64_2> comm_vec;

            for(auto & [id,pdat] : sched.patch_data.owned_data){
                if(pdat.pos_s.size() > 0){

                    sycl::buffer<u64> & newid = newid_buf_map.at(id);

                    if(true){

                        auto nid = newid.get_access<sycl::access::mode::read>();
                        
                        std::unordered_map<u64 , std::unique_ptr<PatchData>> send_map;
                        for(u32 i = pdat.pos_s.size()-1 ; i < pdat.pos_s.size() ; i--){
                            if(id != nid[i]){
                                std::cout << id  << " " << i << " " << nid[i] << "\n";
                                std::unique_ptr<PatchData> & pdat_int = send_map[nid[i]];

                                if(! pdat_int){
                                    pdat_int = std::make_unique<PatchData>();
                                }

                                pdat.extract_particle(i, pdat_int->pos_s, pdat_int->pos_d, pdat_int->U1_s, pdat_int->U1_d, pdat_int->U3_s, pdat_int->U3_d);
                            }
                                
                        }std::cout << std::endl;

                        for(auto & [receiver_pid, pdat_ptr] : send_map){
                            //std::cout << "send " << id << " -> " << receiver_pid <<  " len : " << pdat_ptr->pos_s.size()<<std::endl;


                            comm_vec.push_back(u64_2{sched.patch_list.id_patch_to_global_idx[id],sched.patch_list.id_patch_to_global_idx[receiver_pid]});
                            comm_pdat.push_back(std::move(pdat_ptr));

                        }
                    }
                }
            }

            
            std::unordered_map<u64, std::vector<std::tuple<u64, std::unique_ptr<PatchData>>>> part_xchg_map;
            for(u32 i = 0; i < comm_pdat.size(); i++){
                
                //std::cout << comm_vec[i].x() << " " << comm_vec[i].y() << " " << comm_pdat[i].get() << std::endl; 
            }

            patch_data_exchange_object(
                sched.patch_list.global, 
                comm_pdat, comm_vec, 
                part_xchg_map);

            for(auto & [recv_id, vec_r] : part_xchg_map){
                //std::cout << "patch " << recv_id << "\n";
                for(auto & [send_id, pdat] : vec_r){
                    //std::cout << "    " << send_id << " len : " << pdat->pos_s.size() << "\n"; 

                    PatchData & pdat_recv = sched.patch_data.owned_data[recv_id];

                    //*
                    pdat_recv.insert_particles(
                        pdat->pos_s,
                        pdat->pos_d,
                        pdat->U1_s,
                        pdat->U1_d,
                        pdat->U3_s,
                        pdat->U3_d);
                        //*/
                }
            }

            dump_state("step"+std::to_string(stepi)+"/",sched);
        }

        // TODO test if a interface of size 0.5x0.5x0.5 exist == error
    }

    // std::cout << sched.dump_status() << std::endl;

    std::cout << "changing crit\n";
    sched.crit_patch_merge = 30;
    sched.crit_patch_split = 100;
    sched.scheduler_step(true, true);

    // std::cout << sched.dump_status() << std::endl;
    //*/

    printf("shoudl resize : %d",sched.should_resize_box(mpi_handler::world_rank >4 ));

    sched.free_mpi_required_types();

    mpi_handler::close();

}