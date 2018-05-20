# Copyright (c) 2015-2017 Wind River Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#


def qemuMonitorCommand(domain, cmd, flags):
    retval = '{"return":[{"current":true,"CPU":0,"pc":-2130513274,' \
             '"halted":true,"thread_id":86064},' \
             '{"current":false,"CPU":1,"pc":-2130513274,' \
             '"halted":true,"thread_id":86065},' \
             '{"current":false,"CPU":2,"pc":-2130513274,' \
             '"halted":true,"thread_id":86066},' \
             '{"current":false,"CPU":3,"pc":-2130513274,' \
             '"halted":true,"thread_id":86067}],"id":"libvirt-156"}'
    return retval
