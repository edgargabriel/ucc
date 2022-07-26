/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef FANOUT_H_
#define FANOUT_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_FANOUT_ALG_KNOMIAL,
    UCC_TL_UCP_FANOUT_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_fanout_algs[UCC_TL_UCP_FANOUT_ALG_LAST + 1];

ucc_status_t ucc_tl_ucp_fanout_init(ucc_tl_ucp_task_t *task);

#endif
