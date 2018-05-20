#
# cgcs_messaging.py, this forwards messages between nova-compute and guests
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2013-2016 Wind River Systems, Inc.
#

import nova.context
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import socket

from nova import utils

LOG = logging.getLogger(__name__)

# unix socket address we bind to
server_grp_name = 'cgcs.server_grp'


# The server group message is JSON encoded null-terminated string without
# embedded newlines.

# message_keys
GRP_DATA = "data"
GRP_DEST_ADDR = "dest_addr"
GRP_DEST_INSTANCE = "dest_instance"
GRP_LOG_MSG = "log_msg"
GRP_MSG_TYPE = "msg_type"
GRP_ORIG_MSG_TYPE = "orig_msg_type"
GRP_SEQ = "seq"
GRP_SOURCE_ADDR = "source_addr"
GRP_SOURCE_INSTANCE = "source_instance"
GRP_VERSION = "version"

# corresponds to server_group_msg_type
GRP_BROADCAST = "broadcast"
GRP_NACK = "nack"
GRP_NOTIFICATION = "notification"
GRP_STATUS_QUERY = "status_query"
GRP_STATUS_RESP = "status_response"
GRP_STATUS_RESP_DONE = "status_response_done"
GRP_UNKNOWN = "unknown"

# currently only support one version
GRP_CURRENT_VERSION = 2

# current max message size
GRP_MAX_MSGLEN = 4096


class CGCSMessaging(object):
    def __init__(self, compute_task_api):
        self._do_setup(compute_task_api)

    # this as a helper makes it easier to mock out for testing
    def _do_setup(self, compute_task_api):
        self.server_grp_sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        server_grp_address = '\00' + server_grp_name
        self.server_grp_sock.bind(server_grp_address)
        nova.utils.spawn_n(self.handle_server_grp_msg, compute_task_api)

    @staticmethod
    def instance_name_to_id(instance_name):
        """Get the instance ID.  instance_name should look like
           "instance-xxxxxxxx" where xxxxxxxx is the instance id in the
           database, and is in hex.
        """
        # strip off the "instance-" at the front
        instance_id_str = instance_name.partition('-')[2]
        # strip off anything after the trailing null
        instance_id_str = instance_id_str.partition(b'\0')[0]
        # convert to int assuming it's in hex
        instance_id = int(instance_id_str, base=16)
        return instance_id

    def broadcast_server_grp_msg(self, context, compute_task_api,
                                 s_instance, data):
        """Send message coming from the cgcs messaging backchannel daemon
           (originating from up in the guest) to all other servers in server
           group.
        """
        outmsg = dict()
        outmsg[GRP_VERSION] = GRP_VERSION
        outmsg[GRP_MSG_TYPE] = GRP_BROADCAST
        outmsg[GRP_SOURCE_INSTANCE] = s_instance
        outmsg[GRP_DATA] = data

        LOG.debug("sending broadcast msg %s", repr(outmsg))

        # get the instance ID.
        instance_id = self.instance_name_to_id(s_instance)

        # Send to nova-conductor where we can do database lookups directly
        compute_task_api.send_server_group_msg(context, outmsg,
                                               instance_id=instance_id)

    def server_grp_status_query(self, context, compute_task_api, s_instance,
                                seqnum):
        """The specified instance wants the current status of all the other
           instances in the server group.
        """
        # get the instance ID.
        instance_id = self.instance_name_to_id(s_instance)
        # Send to nova-conductor where we can do database lookups directly
        statuslist = compute_task_api.get_server_group_status(context,
                                                              instance_id)
        for status in statuslist:
            outmsg = dict()
            outmsg[GRP_VERSION] = GRP_CURRENT_VERSION
            outmsg[GRP_MSG_TYPE] = GRP_STATUS_RESP
            outmsg[GRP_SEQ] = seqnum
            outmsg[GRP_DATA] = status
            LOG.debug("sending group_status_query msg %s", repr(outmsg))

            # send the message to the requesting instance
            self.send_server_grp_msg(s_instance, outmsg)

        # Send "status resp complete" message
        outmsg = dict()
        outmsg[GRP_VERSION] = GRP_CURRENT_VERSION
        outmsg[GRP_MSG_TYPE] = GRP_STATUS_RESP_DONE
        outmsg[GRP_SEQ] = seqnum
        outmsg[GRP_DATA] = status
        LOG.debug("sending group_status_query_done msg %s", repr(outmsg))

        self.send_server_grp_msg(s_instance, outmsg)

    def send_server_grp_nack(self, s_instance, msg_type, log_msg):
        """Sends a nack to guest to indicate parsing or protocol failure when
           handling messages from guest.
        """
        outmsg = dict()
        outmsg[GRP_VERSION] = GRP_CURRENT_VERSION
        outmsg[GRP_MSG_TYPE] = GRP_NACK
        outmsg[GRP_ORIG_MSG_TYPE] = msg_type
        outmsg[GRP_LOG_MSG] = log_msg
        LOG.debug("sending nack msg %s", repr(outmsg))

        # send the message to the requesting instance
        self.send_server_grp_msg(s_instance, outmsg)

    def handle_server_grp_msg(self, compute_task_api):
        """Receive messages coming from the cgcs messaging backchannel daemon
           (originating from up in the guest).
        """
        context = nova.context.get_admin_context()

        while True:
            inmsg = self.server_grp_sock.recv(GRP_MAX_MSGLEN)
            if not inmsg:
                continue
            LOG.debug("got datagram %s", repr(inmsg))

            # decode the message
            inmsg = jsonutils.loads(inmsg)

            s_instance = inmsg.get('source_instance')

            version = inmsg.get('version')
            if version != GRP_CURRENT_VERSION:
                log_msg = "invalid guest agent message version %d, " \
                          "expecting %d" % (version, GRP_CURRENT_VERSION)
                self.send_server_grp_nack(s_instance, GRP_UNKNOWN, log_msg)
                continue

            data = inmsg.get(GRP_DATA)
            if data is None:
                log_msg = "no valid data found in the message"
                self.send_server_grp_nack(s_instance, GRP_UNKNOWN, log_msg)
                continue

            msgtype = data.get(GRP_MSG_TYPE, GRP_UNKNOWN)

            version = data.get(GRP_VERSION)
            if version != GRP_CURRENT_VERSION:
                log_msg = "invalid server group message version %d, " \
                          "expecting %d" % (version, GRP_CURRENT_VERSION)
                self.send_server_grp_nack(s_instance, msgtype, log_msg)
                continue

            if msgtype == GRP_BROADCAST:
                msg_data = data.get(GRP_DATA)
                nova.utils.spawn_n(self.broadcast_server_grp_msg,
                                    context, compute_task_api, s_instance,
                                    msg_data)

            elif msgtype == GRP_STATUS_QUERY:
                seqnum = data.get(GRP_SEQ)
                nova.utils.spawn_n(self.server_grp_status_query,
                                    context, compute_task_api, s_instance,
                                    seqnum)
            else:
                log_msg = "got server group message with invalid type %d" \
                          % msgtype
                self.send_server_grp_nack(s_instance, msgtype, log_msg)

    @utils.synchronized(server_grp_name)
    def send_server_grp_msg(self, instance_name, data):
        """send message to the guest via the cgcs messaging
           backchannel daemon
        """
        cgcs_msg_address = '\00' + 'cgcs.messaging'

        # add app-to-daemon header onto the message
        # need to strip trailing NULs off the name otherwise LOG chokes
        LOG.debug("sending to %s ", instance_name.strip('\x00'))

        outmsg = dict()
        outmsg[GRP_VERSION] = GRP_CURRENT_VERSION
        outmsg[GRP_DEST_INSTANCE] = str(instance_name)
        outmsg[GRP_DEST_ADDR] = server_grp_name
        outmsg[GRP_DATA] = data
        outmsg = jsonutils.dumps(outmsg, separators=(',', ':'))

        LOG.debug("sending out to cgcs.messaging %s", outmsg)

        # Send it to the guest via the host agent unix socket
        self.server_grp_sock.sendto(outmsg, socket.MSG_DONTWAIT,
                                    cgcs_msg_address)

    @staticmethod
    def build_notification_msg(data):
        """Take the given data, prepend it with the header for a server group
           notification message.
        """
        outmsg = dict()
        outmsg[GRP_VERSION] = GRP_CURRENT_VERSION
        outmsg[GRP_MSG_TYPE] = GRP_NOTIFICATION
        outmsg[GRP_DATA] = data
        return outmsg


def send_server_grp_notification(context, event_type, payload, instance_uuid):
    """Sends a notification to other instances in the same server group"""
    message = {}
    message['payload'] = jsonutils.to_primitive(payload,
                                                convert_instances=True)
    message['event_type'] = event_type
    message['priority'] = 'INFO'
    message['timestamp'] = timeutils.utcnow().isoformat()

    # Do whatever is needed to make a message ready to send
    message = CGCSMessaging.build_notification_msg(message)
    LOG.debug("sending notification msg %s", repr(message))

    from nova import conductor
    task_api = conductor.ComputeTaskAPI()
    task_api.send_server_group_msg(context, message,
                                   instance_uuid=instance_uuid)
