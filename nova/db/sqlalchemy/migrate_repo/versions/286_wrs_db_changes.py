#
# Copyright (c) 2013-2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#

from sqlalchemy import Table, MetaData, Column, Integer

# WRS Database changes
# -- add per-instance min_vcpus, max_vcpus


def upgrade(migrate_engine):
    meta = MetaData()
    meta.bind = migrate_engine

    # Adding min_vcpus/max_vcpus to instances table
    instances = Table('instances', meta, autoload=True)
    min_vcpus = Column('min_vcpus', Integer())
    max_vcpus = Column('max_vcpus', Integer())
    if not hasattr(instances.c, 'min_vcpus'):
        instances.create_column(min_vcpus)
    if not hasattr(instances.c, 'max_vcpus'):
        instances.create_column(max_vcpus)
    shadow_instances = Table('shadow_instances', meta, autoload=True)
    if not hasattr(shadow_instances.c, 'min_vcpus'):
        shadow_instances.create_column(min_vcpus.copy())
    if not hasattr(shadow_instances.c, 'max_vcpus'):
        shadow_instances.create_column(max_vcpus.copy())
