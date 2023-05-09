# Contributing

## Git Workflow

### Branches
The project is divided up into three main branch areas: `main`, `stable` and `feature`. 
- `main` reflects the current version of the project.
- `stable` reflects the current stable features in the project.
- `feature` reflects a feature in progress. Named in the form: `<IssueNr>-<FeatureName>`, where spaces are replaced with "-". New feature branches are based on the `stable` branch or other feature branches. Most often, these branches are restricted to a few files not being worked upon in other branches to avoid conflicts.

Additionally, there is the `hotfix` branch used for quick fixes of the `main` branch. New hotfix branches are based on the `main` branch as a result. 

### Commits
- Written in the **imperative form**, e.g. "Fix failing burger" where the subject line is *below* 50 characters and starts with a capital character. It is encouraged to restrict the scope of the commit such that it can be easily summarized. I.e. do not change a typo in one file and create a new module in another in the same commit. 
- If the commit is aimed to solve a specific issue, put a reference to the issue at the bottom, e.g. "Resolves: #42".
- Include a description in the body if necessary.

### Issues
Keeps track of what is being done and what needs to be done. Issues both reflect discovered flaws and features in progress. If the issue relates to a specific project, it is encouraged to add it to a related project board.
Include steps for reproducing issues.

When working on an issue you **must** assign yourself to it as to prevent miscommunication and conflicting work!

### Merging and Rebasing
Using both merging and rebasing is beneficial in order to maintain a clean commit history throughout the project. It is important to know when to use what.

- `merge` is used when the target branch is public and meant to be shared. This is for example when new features are brought into the `main` branch. It is then encouraged to squash (--squash flag) the commit in order to keep the commit history of such branches clean.
- `rebase` is used for private branches. This is for example when a `feature` is based on another and can be reincorporated back into the original, as two commit histories may prove unnecessary. This gives the benefit of only having one `merge` in the end instead of two. **NOTE:** never rebase a public branch onto a private target as it leads to diverging commit histories!

### Pull Requests
Once a feature is completed, a pull request of the feature branch should be made to merge with the `stable` branch. These ***require*** peer review in order to be accepted. After being accepted, the corresponding issue should be closed and the feature branch deleted to avoid deprecated branching.
**NOTE:** Merges should also in the majority of cases be squashed to avoid clutter in the commit history.

### Continuous Integration
All pushed commits to the remote branch go through C.I. testing. Make sure to take note if the tests do not pass and always fix them if a pull request is intended. Failing merges will not be accepted!

## Naming conventions
[RFC 430](https://github.com/rust-lang/rfcs/blob/master/text/0430-finalizing-naming-conventions.md) desribes the Rust naming conventions used. 

## Testing
Each feature requires testing in order to ascertain functionality. 
Placement of tests adheres to the common practices of Rust where:

- **Unit tests** are placed as their own module in the file of which the tested functionality is defined.
- **Integration tests** are placed as their own file or module under the directory ``tests/``.
