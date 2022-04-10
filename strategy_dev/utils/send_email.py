#!/usr/bin/python
# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.header import Header


class AlertEmailer():

    __instance = None

    @staticmethod
    def getInstance():
        """Static access method"""
        if AlertEmailer.__instance is None:
            AlertEmailer()
        return AlertEmailer.__instance

    def __init__(self):
        if AlertEmailer.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            AlertEmailer.__instance = self

            self.mail_host = None
            self.mail_user = None
            self.mail_pass = None
            self.smtp_port = None

    def set_parameter(self, host, user, passwd, port):
        self.mail_host = host
        self.mail_user = user
        self.mail_pass = passwd
        self.smtp_port = port

    def set_sender_receiver(self, sender, receivers):
        self.sender = sender
        self.receivers = receivers

    def send_email_alert(self, message):

        if not self.smtp_port or \
                not self.mail_host or \
                not self.mail_pass or \
                not self.mail_user:
            raise Exception("emailer parameter not set")

        if not self.sender or \
                not self.receivers:
            raise Exception("emailer sender/receivers not set")

        try:
            message = MIMEText(message, 'plain', 'utf-8')
            message['From'] = Header("交易实盘", 'utf-8')
            message['To'] = Header("AlgoTrading Group", 'utf-8')
            message['Subject'] = Header('交易提醒', 'utf-8')

            smtpObj = smtplib.SMTP_SSL(self.mail_host, self.smtp_port)
            smtpObj.ehlo()
            smtpObj.login(self.mail_user, self.mail_pass)
            smtpObj.sendmail(self.sender, self.receivers, message.as_string())

        except smtplib.SMTPException as e:
            print(f"{e}")
            print(f"Error: 无法发送邮件")


