from paver.easy import task, needs
from paver.setuputils import setup, install_distutils_tasks

import sys
sys.path.insert(0, '.')
import version

install_distutils_tasks()


setup(name='colonists',
      version=version.getVersion(),
      description='Board game played on a hexagonal-grid.',
      keywords='game hex',
      author='Christian Fobel',
      author_email='christian@fobel.net',
      url='https://github.com/cfobel/colonists',
      license='GPL',
      install_requires=['pandas', 'numpy', 'matplotlib'],
      include_package_data=True,
      packages=['colonists'])


@task
@needs('generate_setup', 'minilib', 'setuptools.command.sdist')
def sdist():
    """Overrides sdist to make sure that our setup.py is generated."""
    pass
