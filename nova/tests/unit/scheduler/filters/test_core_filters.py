#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#
# Copyright (c) 2016-2017 Wind River Systems, Inc.
#

# import mock

from nova import objects
from nova.scheduler.filters import core_filter
from nova import test
from nova.tests.unit.scheduler import fakes

FLAVOR_1 = objects.Flavor(vcpus=1, extra_specs={})
FLAVOR_2 = objects.Flavor(vcpus=2, extra_specs={})
IMAGE_PROPS = objects.ImageMeta(properties=objects.ImageMetaProps())


class TestCoreFilter(test.NoDBTestCase):

    def test_core_filter_passes(self):
        self.filt_cls = core_filter.CoreFilter()
        spec_obj = objects.RequestSpec(flavor=objects.Flavor(vcpus=1,
                                                             extra_specs={}),
                                       image=IMAGE_PROPS)
        cpu_allocation_ratio = 2
        host = fakes.FakeHostState('host1', 'node1',
                {'vcpus_total': 4, 'vcpus_used': 7 / cpu_allocation_ratio,
                 'cpu_allocation_ratio': cpu_allocation_ratio})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_core_filter_fails_safe(self):
        self.filt_cls = core_filter.CoreFilter()
        spec_obj = objects.RequestSpec(flavor=objects.Flavor(vcpus=1,
                                                             extra_specs={}),
                                       image=IMAGE_PROPS)
        host = fakes.FakeHostState('host1', 'node1', {})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_core_filter_fails(self):
        self.filt_cls = core_filter.CoreFilter()
        spec_obj = objects.RequestSpec(flavor=objects.Flavor(vcpus=1,
                                                             extra_specs={}),
                                       image=IMAGE_PROPS)
        cpu_allocation_ratio = 2
        host = fakes.FakeHostState('host1', 'node1',
                {'vcpus_total': 4, 'vcpus_used': 8 / cpu_allocation_ratio,
                 'cpu_allocation_ratio': cpu_allocation_ratio})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_core_filter_single_instance_overcommit_fails(self):
        self.filt_cls = core_filter.CoreFilter()
        spec_obj = objects.RequestSpec(flavor=objects.Flavor(vcpus=2,
                                                             extra_specs={}),
                                       image=IMAGE_PROPS)
        host = fakes.FakeHostState('host1', 'node1',
                {'vcpus_total': 1, 'vcpus_used': 0,
                 'cpu_allocation_ratio': 2})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    # WRS: we do not support AggregateCoreFilter since we report the
    # "vcpus_used" value with the cpu_allocation_ratio already factored-in.
    # This makes it difficult to account for per-aggregate ratios.
    #
    # @mock.patch('nova.scheduler.filters.utils.aggregate_values_from_key')
    # def test_aggregate_core_filter_value_error(self, agg_mock):
    #     self.filt_cls = core_filter.AggregateCoreFilter()
    #     spec_obj = objects.RequestSpec(
    #         context=mock.sentinel.ctx, flavor=objects.Flavor(vcpus=1,
    #                                                          extra_specs={}))
    #     host = fakes.FakeHostState('host1', 'node1',
    #             {'vcpus_total': 4, 'vcpus_used': 7,
    #              'cpu_allocation_ratio': 2})
    #     agg_mock.return_value = set(['XXX'])
    #     self.assertTrue(self.filt_cls.host_passes(host, spec_obj))
    #     agg_mock.assert_called_once_with(host, 'cpu_allocation_ratio')
    #     self.assertEqual(4 * 2, host.limits['vcpu'])
    #
    # @mock.patch('nova.scheduler.filters.utils.aggregate_values_from_key')
    # def test_aggregate_core_filter_default_value(self, agg_mock):
    #     self.filt_cls = core_filter.AggregateCoreFilter()
    #     spec_obj = objects.RequestSpec(
    #         context=mock.sentinel.ctx, flavor=objects.Flavor(vcpus=1))
    #     host = fakes.FakeHostState('host1', 'node1',
    #             {'vcpus_total': 4, 'vcpus_used': 8,
    #              'cpu_allocation_ratio': 2})
    #     agg_mock.return_value = set([])
    #     # False: fallback to default flag w/o aggregates
    #     self.assertFalse(self.filt_cls.host_passes(host, spec_obj))
    #     agg_mock.assert_called_once_with(host, 'cpu_allocation_ratio')
    #     # True: use ratio from aggregates
    #     agg_mock.return_value = set(['3'])
    #     self.assertTrue(self.filt_cls.host_passes(host, spec_obj))
    #     self.assertEqual(4 * 3, host.limits['vcpu'])
    #
    # @mock.patch('nova.scheduler.filters.utils.aggregate_values_from_key')
    # def test_aggregate_core_filter_conflict_values(self, agg_mock):
    #     self.filt_cls = core_filter.AggregateCoreFilter()
    #     spec_obj = objects.RequestSpec(
    #         context=mock.sentinel.ctx, flavor=objects.Flavor(vcpus=1))
    #     host = fakes.FakeHostState('host1', 'node1',
    #             {'vcpus_total': 4, 'vcpus_used': 8,
    #              'cpu_allocation_ratio': 1})
    #     agg_mock.return_value = set(['2', '3'])
    #     # use the minimum ratio from aggregates
    #     self.assertFalse(self.filt_cls.host_passes(host, spec_obj))
    #     self.assertEqual(4 * 2, host.limits['vcpu'])
