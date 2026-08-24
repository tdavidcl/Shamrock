
#include "shambase/WithUUID.hpp"
#include "shambase/aliases_int.hpp"
namespace shamrock::solvergraph {

    template<typename T>
    class LifetimeTracker : public shambase::WithUUID<LifetimeTracker<T>, u64> {
        public:
        /// Called when a tracked object is created
        inline static void (*on_create)(u64 uuid) = nullptr;
        /// Called when a tracked object is destroyed
        inline static void (*on_destroy)(u64 uuid) = nullptr;

        /// Called when the state of a tracked object changes (e.g. edges are rebound)
        inline static void (*on_state_update)(T &object) = nullptr;
        /// Called when an operation is performed on a tracked object (e.g. evaluation)
        inline static void (*on_op)(u64 uuid, u64 op_id) = nullptr;

        public:
        LifetimeTracker();
        ~LifetimeTracker();

        void trace_create(u64 uuid);
        void trace_destroy(u64 uuid);
        void trace_state_update(T &object);
        void trace_op(u64 uuid, u64 op_id);
    };
} // namespace shamrock::solvergraph
