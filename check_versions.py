import sys
import pkg_resources

packages = ['numpy', 'torch', 'transformers', 'torchaudio', 'torchvision']
for package in packages:
    try:
        dist = pkg_resources.get_distribution(package)
        print(f"{package}: {dist.version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: Not Installed")
