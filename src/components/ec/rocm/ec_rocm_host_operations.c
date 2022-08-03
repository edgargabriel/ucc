/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm_executor.h"
#include "ec_rocm.h"
#include "utils/ucc_math_op.h"
#include "components/mc/ucc_mc.h"
#include <inttypes.h>


#define ROCM_REDUCE_WITH_OP_DEFAULT_HOST(OPNAME, _OP, _TYPE, _ATYPE)            \
   void UCC_ROCM_REDUCE_HOST_##OPNAME##_##_TYPE(hipStream_t stream,             \
                                                hipError_t status,              \
                                                void *rhost_task)               \
    {                                                                           \
        ucc_ec_rocm_executor_interruptible_task_t* task =                       \
          (ucc_ec_rocm_executor_interruptible_task_t *)rhost_task;              \
        ucc_eee_task_reduce_t *tr = &task->super.args.reduce;                   \
                                                                                \
        uint16_t              flags  = task->super.args.flags;                  \
        size_t                count  = tr->count;                               \
        int                   n_srcs = tr->n_srcs;                              \
        const _TYPE         **s      = (const _TYPE **)tr->srcs;                \
        _TYPE *               d      = (_TYPE *)tr->dst;                        \
        size_t                i;                                                \
                                                                                \
        switch (n_srcs) {                                                       \
        case 2:                                                                 \
            for (i = 0; i < count; i++) {                                       \
                d[i] = _OP##_2(s[0][i], s[1][i]);                               \
            }                                                                   \
            break;                                                              \
        case 3:                                                                 \
            for (i = 0; i < count; i++) {                                       \
                d[i] = _OP##_3(s[0][i], s[1][i], s[2][i]);                      \
            }                                                                   \
            break;                                                              \
        case 4:                                                                 \
            for (i = 0; i < count; i++) {                                       \
                d[i] = _OP##_4(s[0][i], s[1][i], s[2][i], s[3][i]);             \
            }                                                                   \
            break;                                                              \
        default:                                                                \
            for (i = 0; i < count; i++) {                                       \
                d[i] = _OP(s[0][i], s[1][i]);                                   \
                for (size_t j = 2; j < n_srcs; j++) {                           \
                    d[i] = _OP(d[i], s[j][i]);                                  \
                }                                                               \
            }                                                                   \
            break;                                                              \
        }                                                                       \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                      \
            for (i = 0; i < count; i++) {                                       \
                d[i] = d[i] * (_ATYPE)tr->alpha;                                \
            }                                                                   \
        }                                                                       \
        ucc_mpool_put(task);                                                    \
    }

#define ROCM_REDUCE_WITH_OP_STRIDED_HOST(OPNAME, _OP, _TYPE, _ATYPE)           \
  void UCC_ROCM_REDUCE_HOST_STRIDED_##OPNAME##_##_TYPE(hipStream_t stream,     \
                                                       hipError_t status,      \
                                                       void *rhost_task)       \
    {                                                                          \
        ucc_ec_rocm_executor_interruptible_task_t* task =                      \
          (ucc_ec_rocm_executor_interruptible_task_t *)rhost_task;             \
        ucc_eee_task_reduce_strided_t *trs = &task->super.args.reduce_strided; \
                                                                               \
        const _TYPE *s1       = (const _TYPE *)trs->src1;                      \
        const _TYPE *s2       = (const _TYPE *)trs->src2;                      \
        _TYPE        *d       = (_TYPE *)trs->dst;                             \
        size_t        count   = trs->count;                                    \
        size_t        stride  = trs->stride;                                   \
        uint16_t      n_src2  = trs->n_src2;                                   \
        uint16_t      flags   = task->super.args.flags;                        \
        const double  alpha   = trs->alpha;                                    \
                                                                               \
        size_t ld    = stride / sizeof(_TYPE);                                 \
        size_t i;                                                              \
                                                                               \
        ucc_assert(stride % sizeof(_TYPE) == 0);                               \
        switch (n_src2) {                                                      \
        case 1:                                                                \
            for (i = 0; i < count; i++) {                                      \
                d[i] = _OP##_2(s1[i], s2[i]);                                  \
            }                                                                  \
            break;                                                             \
        case 2:                                                                \
            for (i = 0; i < count; i++) {                                      \
                d[i] = _OP##_3(s1[i], s2[i], s2[i + ld]);                      \
            }                                                                  \
            break;                                                             \
        case 3:                                                                \
            for (i = 0; i < count; i++) {                                      \
                d[i] = _OP##_4(s1[i], s2[i], s2[i + ld], s2[i + 2 * ld]);      \
            }                                                                  \
            break;                                                             \
        default:                                                               \
            for (i = 0; i < count; i++) {                                      \
                d[i] = _OP(s1[i], s2[i]);                                      \
                for (size_t j = 1; j < n_src2; j++) {                          \
                    d[i] = _OP(d[i], s2[i + j * ld]);                          \
                }                                                              \
            }                                                                  \
            break;                                                             \
        }                                                                      \
        if (flags & UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA) {                     \
            for (i = 0; i < count; i++) {                                      \
                d[i] = d[i] * (_ATYPE)alpha;                                   \
            }                                                                  \
        }                                                                      \
        ucc_mpool_put(task);                                                   \
   }

#define CREATE_HOST_FUNCTIONS_INT(_OP, _DO_OP)                                 \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, int8_t,   int8_t   ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, int16_t,  int16_t  ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, int32_t,  int32_t  ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, int64_t,  int64_t  ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, uint8_t,  uint8_t  ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, uint16_t, uint16_t ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, uint32_t, uint32_t ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, uint64_t, uint64_t );

#define CREATE_HOST_FUNCTIONS_FLOAT(_OP, _DO_OP)                       \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, float,    float    ); \
  ROCM_REDUCE_WITH_OP_DEFAULT_HOST(_OP,  _DO_OP, double,   double   );


CREATE_HOST_FUNCTIONS_INT(SUM,  DO_OP_PROD);
CREATE_HOST_FUNCTIONS_INT(PROD, DO_OP_PROD);
CREATE_HOST_FUNCTIONS_INT(MIN,  DO_OP_MIN);
CREATE_HOST_FUNCTIONS_INT(MAX,  DO_OP_MAX);
CREATE_HOST_FUNCTIONS_INT(LAND, DO_OP_LAND);
CREATE_HOST_FUNCTIONS_INT(LOR,  DO_OP_LOR);
CREATE_HOST_FUNCTIONS_INT(LXOR, DO_OP_LXOR);
CREATE_HOST_FUNCTIONS_INT(BAND, DO_OP_BAND);
CREATE_HOST_FUNCTIONS_INT(BOR,  DO_OP_BOR);
CREATE_HOST_FUNCTIONS_INT(BXOR, DO_OP_BXOR);

CREATE_HOST_FUNCTIONS_FLOAT(SUM, DO_OP_PROD);
CREATE_HOST_FUNCTIONS_FLOAT(PROD, DO_OP_PROD);
CREATE_HOST_FUNCTIONS_FLOAT(MIN,  DO_OP_MIN);
CREATE_HOST_FUNCTIONS_FLOAT(MAX,  DO_OP_MAX);

#define CREATE_STRIDED_HOST_FUNCTIONS_INT(_OP, _DO_OP)                    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, int8_t,   int8_t   );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, int16_t,  int16_t  );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, int32_t,  int32_t  );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, int64_t,  int64_t  );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, uint8_t,  uint8_t  );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, uint16_t, uint16_t );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, uint32_t, uint32_t );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, uint64_t, uint64_t );

#define CREATE_STRIDED_HOST_FUNCTIONS_FLOAT(_OP, _DO_OP)                  \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, float,    float    );    \
  ROCM_REDUCE_WITH_OP_STRIDED_HOST(_OP,  _DO_OP, double,   double   );

CREATE_STRIDED_HOST_FUNCTIONS_INT(SUM,  DO_OP_SUM);
CREATE_STRIDED_HOST_FUNCTIONS_INT(PROD, DO_OP_PROD);
CREATE_STRIDED_HOST_FUNCTIONS_INT(MIN,  DO_OP_MIN);
CREATE_STRIDED_HOST_FUNCTIONS_INT(MAX,  DO_OP_MAX);
CREATE_STRIDED_HOST_FUNCTIONS_INT(LAND, DO_OP_LAND);
CREATE_STRIDED_HOST_FUNCTIONS_INT(LOR,  DO_OP_LOR);
CREATE_STRIDED_HOST_FUNCTIONS_INT(LXOR, DO_OP_LXOR);
CREATE_STRIDED_HOST_FUNCTIONS_INT(BAND, DO_OP_BAND);
CREATE_STRIDED_HOST_FUNCTIONS_INT(BOR,  DO_OP_BOR);
CREATE_STRIDED_HOST_FUNCTIONS_INT(BXOR, DO_OP_BXOR);

CREATE_STRIDED_HOST_FUNCTIONS_FLOAT(SUM,  DO_OP_SUM);
CREATE_STRIDED_HOST_FUNCTIONS_FLOAT(PROD, DO_OP_PROD);
CREATE_STRIDED_HOST_FUNCTIONS_FLOAT(MIN,  DO_OP_MIN);
CREATE_STRIDED_HOST_FUNCTIONS_FLOAT(MAX,  DO_OP_MAX);

#define LAUNCH_KERNEL(NAME, TYPE, _task, s)                                             \
  do {                                                                                  \
        if (_task->super.args.task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {               \
            hipStreamAddCallback(s, UCC_ROCM_REDUCE_HOST_##NAME##_##TYPE, _task, 0);    \
        } else {                                                                        \
            hipStreamAddCallback(s, UCC_ROCM_REDUCE_HOST_STRIDED_##NAME##_##TYPE,       \
                                 _task, 0);                                             \
        }                                                                               \
    } while (0)

#define DT_REDUCE_INT(type, _task, _op, s)             \
    do {                                               \
        switch (_op) {                                 \
        case UCC_OP_AVG:                               \
        case UCC_OP_SUM:                               \
            LAUNCH_KERNEL(SUM, type, _task, s);        \
            break;                                     \
        case UCC_OP_PROD:                              \
            LAUNCH_KERNEL(PROD, type, _task, s);       \
            break;                                     \
        case UCC_OP_MIN:                               \
            LAUNCH_KERNEL(MIN, type, _task, s);        \
            break;                                     \
        case UCC_OP_MAX:                               \
            LAUNCH_KERNEL(MAX, type, _task, s);        \
            break;                                     \
        case UCC_OP_LAND:                              \
            LAUNCH_KERNEL(LAND, type, _task, s);       \
            break;                                     \
        case UCC_OP_BAND:                              \
            LAUNCH_KERNEL(BAND, type, _task, s);       \
            break;                                     \
        case UCC_OP_LOR:                               \
            LAUNCH_KERNEL(LOR, type, _task, s);        \
            break;                                     \
        case UCC_OP_BOR:                               \
            LAUNCH_KERNEL(BOR, type, _task, s);        \
            break;                                     \
        case UCC_OP_LXOR:                              \
            LAUNCH_KERNEL(LXOR, type, _task, s);       \
            break;                                     \
        case UCC_OP_BXOR:                              \
            LAUNCH_KERNEL(BXOR, type, _task, s);       \
            break;                                     \
        default:                                       \
            ec_error(&ucc_ec_rocm.super,               \
                     "int dtype does not support "     \
                     "requested reduce op: %s",        \
                     ucc_reduction_op_str(_op));       \
            return UCC_ERR_NOT_SUPPORTED;              \
        }                                              \
    } while (0)

#define DT_REDUCE_FLOAT(type, _task, _op, s)           \
    do {                                               \
        switch (_op) {                                 \
        case UCC_OP_AVG:                               \
        case UCC_OP_SUM:                               \
            LAUNCH_KERNEL(SUM, type, _task, s);        \
            break;                                     \
        case UCC_OP_PROD:                              \
            LAUNCH_KERNEL(PROD, type, _task, s);       \
            break;                                     \
        case UCC_OP_MIN:                               \
            LAUNCH_KERNEL(MIN, type, _task, s);        \
            break;                                     \
        case UCC_OP_MAX:                               \
            LAUNCH_KERNEL(MAX, type, _task, s);        \
            break;                                     \
        default:                                       \
            ec_error(&ucc_ec_rocm.super,               \
                     "float dtype does not support "   \
                     "requested reduce op: %s",        \
                     ucc_reduction_op_str(_op));       \
            return UCC_ERR_NOT_SUPPORTED;              \
        }                                              \
    } while (0)

ucc_status_t ucc_ec_rocm_host_reduce (ucc_ec_rocm_executor_interruptible_task_t* task,
                                      hipStream_t                                stream)
{
    ucc_ee_executor_task_args_t *task_args = &task->super.args;
    ucc_reduction_op_t           op;
    ucc_datatype_t               dt;
    size_t                       count;

    if (task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        dt    = task_args->reduce.dt;
        count = task_args->reduce.count;
        op    = task_args->reduce.op;
    } else {
        ucc_assert(task_args->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED);
        dt    = task_args->reduce_strided.dt;
        count = task_args->reduce_strided.count;
        op    = task_args->reduce_strided.op;
    }

    if (count == 0) {
        return UCC_OK;
    }

    switch (dt) {
    case UCC_DT_INT8:
        DT_REDUCE_INT(int8_t, task, op, stream);
        break;
    case UCC_DT_INT16:
        DT_REDUCE_INT(int16_t, task, op, stream);
        break;
    case UCC_DT_INT32:
        DT_REDUCE_INT(int32_t, task, op, stream);
        break;
    case UCC_DT_INT64:
        DT_REDUCE_INT(int64_t, task, op, stream);
        break;
    case UCC_DT_UINT8:
        DT_REDUCE_INT(uint8_t, task, op, stream);
        break;
    case UCC_DT_UINT16:
        DT_REDUCE_INT(uint16_t, task, op, stream);
        break;
    case UCC_DT_UINT32:
        DT_REDUCE_INT(uint32_t, task, op, stream);
        break;
    case UCC_DT_UINT64:
        DT_REDUCE_INT(uint64_t, task, op, stream);
        break;
    case UCC_DT_FLOAT32:
#if SIZEOF_FLOAT == 4
        DT_REDUCE_FLOAT(float, task, op, stream);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64:
#if SIZEOF_DOUBLE == 8
        DT_REDUCE_FLOAT(double, task, op, stream);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    default:
        ec_error(&ucc_ec_rocm.super, "unsupported reduction type (%s)",
                 ucc_datatype_str(dt));
        return UCC_ERR_NOT_SUPPORTED;
    }
    ROCMCHECK(hipGetLastError());
    return UCC_OK;
}

void ucc_ec_rocm_host_memcpy (hipStream_t stream, hipError_t status, void *args)
{
    ucc_ec_rocm_executor_interruptible_task_t *task =
        (ucc_ec_rocm_executor_interruptible_task_t *) args;
    ucc_ee_executor_task_args_t *task_args = &task->super.args;

    memcpy(task_args->copy.dst, task_args->copy.src, task_args->copy.len);
    ucc_mpool_put(task);
}

void ucc_ec_rocm_host_memcpy_multi (hipStream_t stream, hipError_t status, void *args)
{
    ucc_ec_rocm_executor_interruptible_task_t *task =
      (ucc_ec_rocm_executor_interruptible_task_t *) args;
    ucc_ee_executor_task_args_t *task_args = &task->super.args;

    for (int i = 0; i < task_args->copy_multi.num_vectors; i++) {
        memcpy(task_args->copy_multi.dst[i],
               task_args->copy_multi.src[i],
               task_args->copy_multi.counts[i]);
    }
    ucc_mpool_put(task);
}
