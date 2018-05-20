#
# Copyright (c) 2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

from sqlalchemy import Table, MetaData, Column, Integer

# WRS Database changes for L3 CAT Support
# -- add per-compute-node l3_closids, l3_closids_used


def upgrade(migrate_engine):
    meta = MetaData()
    meta.bind = migrate_engine

    compute_nodes = Table('compute_nodes', meta, autoload=True)
    shadow_compute_nodes = Table('shadow_compute_nodes', meta, autoload=True)

    l3_closids = Column('l3_closids', Integer())
    if not hasattr(compute_nodes.c, 'l3_closids'):
        compute_nodes.create_column(l3_closids)
    if not hasattr(shadow_compute_nodes.c, 'l3_closids'):
        shadow_compute_nodes.create_column(l3_closids.copy())

    l3_closids_used = Column('l3_closids_used', Integer())
    if not hasattr(compute_nodes.c, 'l3_closids_used'):
        compute_nodes.create_column(l3_closids_used)
    if not hasattr(shadow_compute_nodes.c, 'l3_closids_used'):
        shadow_compute_nodes.create_column(l3_closids_used.copy())
