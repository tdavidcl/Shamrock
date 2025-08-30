// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file FixedStack.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Fixed-size stack container for high-performance applications
 *
 * This header provides a stack container with compile-time fixed capacity,
 * designed for performance-critical applications where dynamic memory allocation
 * must be avoided. The stack uses a statically allocated array and manages
 * elements using a cursor-based approach.
 *
 * The FixedStack is particularly useful in GPU kernels, real-time systems,
 * and algorithms where memory allocation overhead is prohibitive, such as
 * tree traversal algorithms or depth-first search operations.
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"

namespace shambase {

    /**
     * @brief Fixed-capacity stack container with compile-time size determination
     *
     * A high-performance stack implementation that uses a statically allocated
     * array for storage. The stack capacity is determined at compile time,
     * eliminating runtime memory allocation overhead. The implementation uses
     * a cursor-based approach where the cursor points to the next available
     * slot in the stack array.
     *
     * The stack grows downward in memory (decreasing indices), with the cursor
     * starting at stack_size (empty stack) and decreasing as elements are pushed.
     * This design choice optimizes for common usage patterns in computational
     * algorithms.
     *
     * @tparam T Element type to store in the stack
     * @tparam stack_size Maximum number of elements the stack can hold
     *
     * @note The stack array is intentionally not zero-initialized in the default
     *       constructor to avoid unnecessary initialization overhead in performance-
     *       critical applications.
     *
     * @code{.cpp}
     * // Example: Basic stack operations
     * shambase::FixedStack<u32, 10> stack;
     *
     * // Push elements
     * stack.push(42);
     * stack.push(17);
     * stack.push(8);
     *
     * // Check if stack has elements
     * while (stack.is_not_empty()) {
     *     u32 value = stack.pop();
     *     // Process value (8, 17, 42 in LIFO order)
     * }
     *
     * // Example: Initialize with a value
     * shambase::FixedStack<u32, 5> initialized_stack(100);
     * // Stack contains one element: 100
     *
     * // Example: Use in tree traversal
     * shambase::FixedStack<u32, 64> node_stack;
     * node_stack.push(root_node_id);
     *
     * while (node_stack.is_not_empty()) {
     *     u32 current_node = node_stack.pop();
     *     // Process current node
     *     // Push child nodes for further processing
     *     for (u32 child : get_children(current_node)) {
     *         node_stack.push(child);
     *     }
     * }
     * @endcode
     */
    template<class T, u32 stack_size>
    struct FixedStack {

        static_assert(stack_size > 0, "FixedStack must have a size greater than 0.");

        /// Storage array for stack elements
        T id_stack[stack_size];
        /// Cursor pointing to the next available slot (stack_size = empty, 0 = full)
        u32 stack_cursor;

        // Note that the stack it self is volontarely not initialized
        // do not add it to the constructor otherwise we may have to pay for zero initialization

        /**
         * @brief Default constructor creating an empty stack
         *
         * Initializes the stack cursor to stack_size, indicating an empty stack.
         * The stack array is intentionally not initialized to avoid unnecessary
         * zero-initialization overhead in performance-critical applications.
         */
        FixedStack() : stack_cursor{stack_size} {}

        /**
         * @brief Constructor creating a stack with one initial element
         *
         * Initializes the stack with a single element and sets the cursor
         * appropriately. The stack will contain one element after construction.
         *
         * @param val Initial value to push onto the stack
         */
        FixedStack(T val) : stack_cursor{stack_size - 1} { id_stack[stack_cursor] = val; }

        /**
         * @brief Check if the stack contains any elements
         *
         * Tests whether the stack is not empty by comparing the cursor position
         * with the stack size. An empty stack has cursor equal to stack_size.
         *
         * @return true if the stack contains at least one element, false if empty
         *
         * @note This method is preferred over an is_empty() method for performance
         *       reasons in typical usage patterns where the condition is used in
         *       while loops for stack processing.
         */
        inline bool is_not_empty() const { return stack_cursor < stack_size; }

        /**
         * @brief Push an element onto the top of the stack
         *
         * Adds a new element to the stack by decreasing the cursor and storing
         * the value at the new cursor position. The stack grows downward in
         * memory (towards lower array indices).
         *
         * @param val Value to push onto the stack
         *
         * @pre The stack must not be full (stack_cursor > 0)
         * @post The stack contains one additional element
         *
         * @throws SHAM_ASSERT failure if the stack is full
         *
         * @note This operation has O(1) time complexity
         */
        inline void push(T val) {

            // FixedStack overflow: cannot push to a full stack.
            SHAM_ASSERT(stack_cursor > 0);

            stack_cursor--;
            id_stack[stack_cursor] = val;
        }

        /**
         * @brief Remove and return the top element from the stack
         *
         * Retrieves the element at the top of the stack (at the cursor position)
         * and increases the cursor to effectively remove it from the stack.
         * The element is returned by value following LIFO (Last In, First Out)
         * semantics.
         *
         * @return The top element of the stack
         *
         * @pre The stack must not be empty (is_not_empty() must return true)
         * @post The stack contains one fewer element
         *
         * @throws SHAM_ASSERT failure if the stack is empty
         *
         * @note This operation has O(1) time complexity. The popped element
         *       remains in the array memory but is no longer considered part
         *       of the stack.
         */
        inline T pop() {

            // FixedStack underflow: cannot pop from an empty stack.
            SHAM_ASSERT(is_not_empty());

            T val = id_stack[stack_cursor];
            stack_cursor++;
            return val;
        }
    };

} // namespace shambase
