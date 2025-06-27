"""
The setup.py file is an essential part of packaging and distributing Python projects.
It contains metadata about the project and dependencies, and provides a way to install the project using pip.
"""

from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
    """
    This function will return the list of requirements
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt') as file:
            lines = file.readlines()
            for line in lines:
                requirement=line.strip()

                if requirement and requirement !='-e .':
                    requirement_lst.append(requirement)
        
    except FileNotFoundError as e:
        print("requirements.txt file not found")

    return requirement_lst

print(get_requirements())
setup(
    name='Network Security Project',
    version='0.0.1',
    author='jagadesh',
    author_email='chillajagadesh68@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)






