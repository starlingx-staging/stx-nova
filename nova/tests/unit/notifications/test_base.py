# Copyright (c) 2017 OpenStack Foundation
# All Rights Reserved.
#
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
import datetime

import mock
from oslo_utils import uuidutils

from nova.compute import cgcs_messaging
from nova.notifications import base
from nova import objects
from nova import test
from nova.tests.unit import fake_instance
from nova import utils


def _get_fake_instance(**kwargs):
    system_metadata = []
    for k, v in kwargs.items():
        system_metadata.append({
            "key": k,
            "value": v
        })

    return {
        "system_metadata": system_metadata,
        "uuid": "uuid",
        "key_data": "ssh-rsa asdf",
        "os_type": "asdf",
    }


class TestNullSafeUtils(test.NoDBTestCase):
    def test_null_safe_isotime(self):
        dt = None
        self.assertEqual('', base.null_safe_isotime(dt))
        dt = datetime.datetime(second=1,
                              minute=1,
                              hour=1,
                              day=1,
                              month=1,
                              year=2017)
        self.assertEqual(utils.strtime(dt), base.null_safe_isotime(dt))

    def test_null_safe_str(self):
        line = None
        self.assertEqual('', base.null_safe_str(line))
        line = 'test'
        self.assertEqual(line, base.null_safe_str(line))


class TestSendInstanceUpdateNotification(test.NoDBTestCase):

    @mock.patch('nova.notifications.objects.base.NotificationBase.emit',
                new_callable=mock.NonCallableMock)  # asserts not called
    # TODO(mriedem): Rather than mock is_enabled, it would be better to
    # configure oslo_messaging_notifications.driver=['noop']
    @mock.patch('nova.rpc.NOTIFIER.is_enabled', return_value=False)
    def test_send_versioned_instance_update_notification_disabled(self,
                                                                  mock_enabled,
                                                                  mock_info):
        """Tests the case that versioned notifications are disabled which makes
        _send_versioned_instance_update_notification a noop.
        """
        base._send_versioned_instance_update(mock.sentinel.ctxt,
                                             mock.sentinel.instance,
                                             mock.sentinel.payload,
                                             mock.sentinel.host,
                                             mock.sentinel.service)

    @mock.patch.object(objects.InstanceGroup, 'get_by_instance_uuid')
    @mock.patch.object(cgcs_messaging, 'send_server_grp_notification')
    @mock.patch.object(base, 'bandwidth_usage')
    @mock.patch.object(base, '_compute_states_payload')
    @mock.patch('nova.rpc.get_notifier')
    @mock.patch.object(base, 'info_from_instance')
    def test_send_legacy_instance_update_notification(self, mock_info,
                                                      mock_get_notifier,
                                                      mock_states,
                                                      mock_bw,
                                                      mock_send_sg_notif,
                                                      mock_get_sg):
        """Tests the case that versioned notifications are disabled and
        assert that this does not prevent sending the unversioned
        instance.update notification.
        """

        self.flags(notification_format='unversioned', group='notifications')

        instance = fake_instance.fake_instance_obj(
            mock.sentinel.ctxt, uuid=uuidutils.generate_uuid())
        base.send_instance_update_notification(mock.sentinel.ctxt,
                                               instance)

        mock_get_notifier.return_value.info.assert_called_once_with(
            mock.sentinel.ctxt, 'compute.instance.update', mock.ANY)
