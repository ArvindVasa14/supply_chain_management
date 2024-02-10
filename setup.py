from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT= '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirememts= []
    with open(file_path) as file_obj:
        requirememts= file_obj.readlines()
        requirememts= [req.replace('/n',' ') for req in requirememts]
        if HYPHEN_E_DOT in requirememts:
            requirememts.remove(HYPHEN_E_DOT)
    return requirememts

setup(
    name='ML Project',
    version='0.0.1',
    author='Aravind',
    author_email='arvindaiml14@gmail.com',
    packages= find_packages(),
    install_requires= get_requirements('requirements.txt')
)