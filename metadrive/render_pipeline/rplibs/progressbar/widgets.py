#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# progressbar  - Text progress bar library for Python.
# Copyright (c) 2005 Nilton Volpato
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
'''Default ProgressBar widgets'''

from __future__ import division

import datetime
import math

try:
    from abc import ABCMeta, abstractmethod
except ImportError:
    AbstractWidget = object
    abstractmethod = lambda fn: fn
else:
    AbstractWidget = ABCMeta('AbstractWidget', (object,), {})


def format_updatable(updatable, pbar):
    if hasattr(updatable, 'update'):
        return updatable.update(pbar)
    else:
        return updatable


class Widget(AbstractWidget):
    '''The base class for all widgets

    The ProgressBar will call the widget's update value when the widget should
    be updated. The widget's size may change between calls, but the widget may
    display incorrectly if the size changes drastically and repeatedly.

    The boolean TIME_SENSITIVE informs the ProgressBar that it should be
    updated more often because it is time sensitive.
    '''

    TIME_SENSITIVE = False

    @abstractmethod
    def update(self, pbar):
        '''Updates the widget.

        pbar - a reference to the calling ProgressBar
        '''


class WidgetHFill(Widget):
    '''The base class for all variable width widgets.

    This widget is much like the \\hfill command in TeX, it will expand to
    fill the line. You can use more than one in the same line, and they will
    all have the same width, and together will fill the line.
    '''

    @abstractmethod
    def update(self, pbar, width):
        '''Updates the widget providing the total width the widget must fill.

        pbar - a reference to the calling ProgressBar
        width - The total width the widget must fill
        '''


class Timer(Widget):
    'Widget which displays the elapsed seconds.'

    TIME_SENSITIVE = True

    def __init__(self, format='Elapsed Time: %s'):
        self.format = format

    @staticmethod
    def format_time(seconds):
        'Formats time as the string "HH:MM:SS".'

        return str(datetime.timedelta(seconds=int(seconds)))

    def update(self, pbar):
        'Updates the widget to show the elapsed time.'

        return self.format % self.format_time(pbar.seconds_elapsed)


class ETA(Timer):
    'Widget which attempts to estimate the time of arrival.'

    TIME_SENSITIVE = True

    def update(self, pbar):
        'Updates the widget to show the ETA or total time when finished.'

        if pbar.currval == 0:
            return 'ETA:  --:--:--'
        elif pbar.finished:
            return 'Time: %s' % self.format_time(pbar.seconds_elapsed)
        else:
            elapsed = pbar.seconds_elapsed
            eta = elapsed * pbar.maxval / pbar.currval - elapsed
            return 'ETA:  %s' % self.format_time(eta)


class FileTransferSpeed(Widget):
    'Widget for showing the transfer speed (useful for file transfers).'

    format = '%6.2f %s%s/s'
    prefixes = ' kMGTPEZY'

    def __init__(self, unit='B'):
        self.unit = unit

    def update(self, pbar):
        'Updates the widget with the current SI prefixed speed.'

        if pbar.seconds_elapsed < 2e-6 or pbar.currval < 2e-6:  # =~ 0
            scaled = power = 0
        else:
            speed = pbar.currval / pbar.seconds_elapsed
            power = int(math.log(speed, 1000))
            scaled = speed / 1000.**power

        return self.format % (scaled, self.prefixes[power], self.unit)


class Rate(Widget):
    'Widget for showing the rate of entries per second'

    def update(self, pbar):
        'Updates the widget with the current SI prefixed speed.'

        if pbar.seconds_elapsed < 2e-6 or pbar.currval < 2e-6:  # =~ 0
            return "Rate: 0 /s"
        else:
            speed = pbar.currval / pbar.seconds_elapsed
            return "Rate: {:4d} /s".format(int(speed))

        return self.format % (scaled, self.prefixes[power], self.unit)


class AnimatedMarker(Widget):
    '''An animated marker for the progress bar which defaults to appear as if
    it were rotating.
    '''

    def __init__(self, markers='|/-\\'):
        self.markers = markers
        self.curmark = -1

    def update(self, pbar):
        '''Updates the widget to show the next marker or the first marker when
        finished'''

        if pbar.finished:
            return self.markers[0]

        self.curmark = (self.curmark + 1) % len(self.markers)
        return self.markers[self.curmark]


# Alias for backwards compatibility
RotatingMarker = AnimatedMarker


class Counter(Widget):
    'Displays the current count'

    def __init__(self, format='%d'):
        self.format = format

    def update(self, pbar):
        return self.format % pbar.currval


class Percentage(Widget):
    'Displays the current percentage as a number with a percent sign.'

    def update(self, pbar):
        return '%3d%%' % pbar.percentage()


class FormatLabel(Timer):
    'Displays a formatted label'

    mapping = {
        'elapsed': ('seconds_elapsed', Timer.format_time),
        'finished': ('finished', None),
        'last_update': ('last_update_time', None),
        'max': ('maxval', None),
        'seconds': ('seconds_elapsed', None),
        'start': ('start_time', None),
        'value': ('currval', None)
    }

    def __init__(self, format):
        self.format = format

    def update(self, pbar):
        context = {}
        for name, (key, transform) in self.mapping.items():
            try:
                value = getattr(pbar, key)

                if transform is None:
                    context[name] = value
                else:
                    context[name] = transform(value)
            except:
                pass

        return self.format % context


class SimpleProgress(Widget):
    'Returns progress as a count of the total (e.g.: "5 of 47")'

    def __init__(self, sep=' of '):
        self.sep = sep

    def update(self, pbar):
        return '%d%s%d' % (pbar.currval, self.sep, pbar.maxval)


class Bar(WidgetHFill):
    'A progress bar which stretches to fill the line.'

    def __init__(self,
                 marker='#',
                 left='|',
                 right='|',
                 fill=' ',
                 fill_left=True):
        '''Creates a customizable progress bar.

        marker - string or updatable object to use as a marker
        left - string or updatable object to use as a left border
        right - string or updatable object to use as a right border
        fill - character to use for the empty part of the progress bar
        fill_left - whether to fill from the left or the right
        '''
        self.marker = marker
        self.left = left
        self.right = right
        self.fill = fill
        self.fill_left = fill_left

    def update(self, pbar, width):
        'Updates the progress bar and its subcomponents'

        left, marker, right = (format_updatable(i, pbar)
                               for i in (self.left, self.marker, self.right))

        width -= len(left) + len(right)
        # Marker must *always* have length of 1
        marker *= int(pbar.currval / pbar.maxval * width)

        if self.fill_left:
            return '%s%s%s' % (left, marker.ljust(width, self.fill), right)
        else:
            return '%s%s%s' % (left, marker.rjust(width, self.fill), right)


class ReverseBar(Bar):
    'A bar which has a marker which bounces from side to side.'

    def __init__(self,
                 marker='#',
                 left='|',
                 right='|',
                 fill=' ',
                 fill_left=False):
        '''Creates a customizable progress bar.

        marker - string or updatable object to use as a marker
        left - string or updatable object to use as a left border
        right - string or updatable object to use as a right border
        fill - character to use for the empty part of the progress bar
        fill_left - whether to fill from the left or the right
        '''
        self.marker = marker
        self.left = left
        self.right = right
        self.fill = fill
        self.fill_left = fill_left


class BouncingBar(Bar):

    def update(self, pbar, width):
        'Updates the progress bar and its subcomponents'

        left, marker, right = (format_updatable(i, pbar)
                               for i in (self.left, self.marker, self.right))

        width -= len(left) + len(right)

        if pbar.finished:
            return '%s%s%s' % (left, width * marker, right)

        position = int(pbar.currval % (width * 2 - 1))
        if position > width:
            position = width * 2 - position
        lpad = self.fill * (position - 1)
        rpad = self.fill * (width - len(marker) - len(lpad))

        # Swap if we want to bounce the other way
        if not self.fill_left:
            rpad, lpad = lpad, rpad

        return '%s%s%s%s%s' % (left, lpad, marker, rpad, right)
