#! /usr/bin/env python

import os

from flask.ext.script import Manager

from app import create_app


app = create_app(os.getenv('APP_CONFIG', 'default'))
manager = Manager(app)

@manager.shell
def make_shell_context():
    return dict(app=app)


if __name__ == '__main__':
    print manager.run
    manager.run()
