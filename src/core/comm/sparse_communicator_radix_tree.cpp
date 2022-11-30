#include "core/tree/radix_tree.hpp"
#include "sparse_communicator.hpp"


template <class u_morton, class vec3> 
struct SparseCommExchanger<Radix_Tree<u_morton, vec3>>{

    static SparseCommResult<Radix_Tree<u_morton, vec3>> sp_xchg(SparsePatchCommunicator & communicator, const SparseCommSource<Radix_Tree<u_morton, vec3>> &send_comm_pdat){

        SparseCommResult<Radix_Tree<u_morton, vec3>> recv_obj;

        if(!send_comm_pdat.empty()){

            std::vector<tree_comm::RadixTreeMPIRequest<u_morton, vec3>> rq_lst;


            auto timer_transfmpi = timings::start_timer("patchdata_exchanger", timings::mpi);

            u64 dtcnt = 0;

            {
                for (u64 i = 0; i < communicator.send_comm_vec.size(); i++) {
                    const Patch &psend = communicator.global_patch_list[communicator.send_comm_vec[i].x()];
                    const Patch &precv = communicator.global_patch_list[communicator.send_comm_vec[i].y()];

                    if (psend.node_owner_id == precv.node_owner_id) {
                        auto & vec = recv_obj[precv.id_patch];
                        dtcnt += send_comm_pdat[i]->memsize();
                        vec.push_back({psend.id_patch, send_comm_pdat[i]->duplicate_to_ptr()});
                    } else {
                        
                        dtcnt += tree_comm::comm_isend(*send_comm_pdat[i], rq_lst, precv.node_owner_id, communicator.local_comm_tag[i], MPI_COMM_WORLD);
                    }
                    
                }
            }

            if (communicator.global_comm_vec.size() > 0) {

                //std::cout << std::endl;
                for (u64 i = 0; i < communicator.global_comm_vec.size(); i++) {

                    const Patch &psend = communicator.global_patch_list[communicator.global_comm_vec[i].x()];
                    const Patch &precv = communicator.global_patch_list[communicator.global_comm_vec[i].y()];

                    if (precv.node_owner_id == mpi_handler::world_rank) {

                        if (psend.node_owner_id != precv.node_owner_id) {
                            
                            recv_obj[precv.id_patch].push_back(
                                {psend.id_patch, std::make_unique<Radix_Tree<u_morton, vec3>>(Radix_Tree<u_morton, vec3>::make_empty())}); // patchdata_irecv(recv_rq, psend.node_owner_id,
                                                                                // global_comm_tag[i], MPI_COMM_WORLD)}
                            tree_comm::comm_irecv_probe(*std::get<1>(recv_obj[precv.id_patch][recv_obj[precv.id_patch].size() - 1]),
                                            rq_lst, psend.node_owner_id, communicator.global_comm_tag[i], MPI_COMM_WORLD);
                        }

                    }
                }
                //std::cout << std::endl;
            }

            tree_comm::wait_all(rq_lst);

            timer_transfmpi.stop(dtcnt);
        communicator.xcgh_byte_cnt += dtcnt;

            //TODO check that this sort is valid
            for(auto & [key,obj] : recv_obj){
                std::sort(obj.begin(), obj.end(),[] (const auto& lhs, const auto& rhs) {
                    return std::get<0>(lhs) < std::get<0>(rhs);
                });
            }



        }

        return std::move(recv_obj);
    }
};







template struct SparseCommExchanger<Radix_Tree<u32, f32_3>>;
template struct SparseCommExchanger<Radix_Tree<u64, f32_3>>;
template struct SparseCommExchanger<Radix_Tree<u32, f64_3>>;
template struct SparseCommExchanger<Radix_Tree<u64, f64_3>>;