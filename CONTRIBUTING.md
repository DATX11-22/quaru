# Contributing

## Git Workflow

### Branches
The project is divided up into three main branch areas: `main`, `dev` and `feature`. 
- `main` reflects the current version of the project.
- `dev` reflects the current stable features in the project.
- `feature` reflects a feature in progress. Named in the form: `<IssueNr>-<FeatureName>`, where spaces are replaced with "-". New feature branches are based on the `dev` branch.

Additionally, there is the `hotfix` branch used for quick fixes of the `main` branch. New hotfix branches are based on the `main` branch as a result. 

### Commits
- Written in the **imperative form**, e.g. "Fix failing burger" where the subject line is *below* 50 characters and starts with a capital character. 
- It is encouraged to restrict the scope of the commit such that it can be easily summarized. I.e. do not change a typo in one file and create a new module in another in the same commit.
- If the commit is aimed to solve a specific issue, put a reference to the issue at the bottom, e.g. "Resolves: #42".

### Issues
Keeps track of what is being done and what needs to be done. Issues both reflect discovered flaws and features in progress. If the issue relates to a specific project, it is encouraged to add it to a related project board.

When working on an issue you **must** assign yourself to it as to prevent miscommunication and conflicting work!

## Naming conventions
[RFC 430](https://github.com/rust-lang/rfcs/blob/master/text/0430-finalizing-naming-conventions.md) desribes the Rust naming conventions used. 

## Testing
Each feature requires testing in order to ascertain functionality. 
Placement of tests adheres to the common practices of Rust where:

- **Unit tests** are placed as their own module in the file of which the tested functionality is defined.
- **Integration tests** are placed as their own file or module under the directory ``tests/``.
