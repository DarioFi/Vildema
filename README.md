# Vildema 

Vildema is a file-based, Machine Learning research-oriented data management system. It provides a flexible way to read experiment 
results from multiple files, even with inconsistent formats and allows for quick and easy querying through a python API. It also further implements
some specific basic data analysis and visualization tools to help researchers quickly analyze their results.

## Design Principles
The design of Vildema is based on the following principles:
1. **Robustness**: Vildema is designed to be file based, append only, so that it can handle partial loads of data (based on file patterns) and partial synchronizations with servers.
2. **Stand-alone runtime**: Vildema is designed to run inside a python process when analyzing data, but it does not require to run when the data is saved, making it compatible with VM environments.
3. **Few Data Points regime**: Vildema is designed to work with a modest number of data points, therefore queries are not optimized and just run linear scans (with some care to ordering and data moving)
4. **Flexibility**: Vildema assumes that each data entry in the database has different fields, reflecting different choices of optimizers or architectures, and therefore it can silently handle missing fields
5. **Machine Learning Research Oriented**: Vildema can manage file fields for checkpoint files and log files with training trajectories.

## Workflow 

## Typical usage

### Todo:
- [ ] Add more documentation and examples
- [ ] Decide how to handle browser configs and settings (columns hidden etc.) persistence between sessions

