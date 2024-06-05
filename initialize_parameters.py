import ext.pyyaml.yaml as yaml

# Parse the YAML content
data = yaml.safe_load("newYaml.yaml")

# Print the parsed data
print(data)