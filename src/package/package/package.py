# -*- coding: utf-8 -*-

""" Basic package file """


class Package:
    """ Basic template class """

    def __init__(self):
        """ Init message object """
        self.message = "Hello World"

    def say_hi(self):
        """ Say hi """
        print(self.message)

    def say_bye(self):
        """ Say bye """
        print("Bye")
