## v0.2.1
## Minor Changes
* OpenLabeling is a git submodule (with poetry virtual env manager)
* Workflow function to run full training (forward+backwards)

## v0.2:
### Breaking Changes
* Classes information is now stored in classes.json (previously was in class_list.txt)
* Git commit has been added to the workflows. Configuration in config.ini required.

### Other Changes
* Classes list is passed to OpenLabeling (derived from classes.json dict)
* Only data remains in wrangling_example.py ('workbook'). Code moved to workflow.py. 
 