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

from nova import objects
from nova.scheduler.filters import vcpu_model_filter
from nova import test
from nova.tests.unit.scheduler import fakes


class TestVCPUModelFilter(test.NoDBTestCase):

    def setUp(self):
        super(TestVCPUModelFilter, self).setUp()
        self.filt_cls = vcpu_model_filter.VCpuModelFilter()

    def test_vcpu_model_not_specified(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(memory_mb=1024, extra_specs={}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1', {})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_vcpu_model_flavor_passes(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Nehalem'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Broadwell"}'})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_vcpu_model_flavor_fails(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Nehalem'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Conroe"}'})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_vcpu_model_image_passes(self):
        props = objects.ImageMetaProps(hw_cpu_model='Nehalem')
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(memory_mb=1024, extra_specs={}),
            image=objects.ImageMeta(properties=props),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Broadwell"}'})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_vcpu_model_image_fails(self):
        props = objects.ImageMetaProps(hw_cpu_model='Nehalem')
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(memory_mb=1024, extra_specs={}),
            image=objects.ImageMeta(properties=props),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Conroe"}'})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_passthrough_vcpu_model_flavor_passes(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Passthrough'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Broadwell", "features": ["vmx"]}'})
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_passthrough_migrate_vcpu_model_flavor_passes(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Passthrough'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['migrating'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Broadwell", '
                             '"features": ["pge", "avx", "vmx"]}'})
        self.stub_out('nova.objects.ComputeNode.get_by_host_and_nodename',
                self._fake_compute_node_get_by_host_and_nodename)
        self.assertTrue(self.filt_cls.host_passes(host, spec_obj))

    def test_passthrough_migrate_vcpu_model_flavor_fails(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Passthrough'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['migrating'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "IvyBridge", '
                             '"features": ["pge", "avx", "vmx"]}'})
        self.stub_out('nova.objects.ComputeNode.get_by_host_and_nodename',
                self._fake_compute_node_get_by_host_and_nodename)
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_passthrough_migrate_vcpu_model_flavor_features_fails(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Passthrough'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['migrating'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Broadwell", '
                             '"features": ["pge", "avx", "vmx", "clflush"]}'})
        self.stub_out('nova.objects.ComputeNode.get_by_host_and_nodename',
                self._fake_compute_node_get_by_host_and_nodename)
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def test_passthrough_migrate_vcpu_model_flavor_kvm_fails(self):
        spec_obj = objects.RequestSpec(
            flavor=objects.Flavor(extra_specs={'hw:cpu_model': 'Passthrough'}),
            image=objects.ImageMeta(properties=objects.ImageMetaProps()),
            scheduler_hints={'task_state': ['scheduling'], 'host': ['host1'],
                             'node': ['node1']})
        host = fakes.FakeHostState('host1', 'node1',
                {'cpu_info': '{"model": "Broadwell", '
                             '"features": ["pge", "avx"]}'})
        self.assertFalse(self.filt_cls.host_passes(host, spec_obj))

    def _fake_compute_node_get_by_host_and_nodename(self, cn, ctx, host, node):
        cpu_info = '{"model": "Broadwell", "features": ["pge", "avx", "vmx"]}'
        compute_node = objects.ComputeNode(cpu_info=cpu_info)
        return compute_node
