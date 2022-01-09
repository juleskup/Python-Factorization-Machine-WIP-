from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Python Factorization Machines',
    url='https://github.com/juleskup/python_factorization_machine',
    author='Jules Kuperminc',
    author_email='jules.kuperminc@ensae.fr',
    # Needed to actually package something
    packages=['measure'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A simple implementation of a factorization machine with Python',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)