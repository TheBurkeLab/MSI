# Github Essentials
## General Comments & Guidelines
1. When new changes are commited to master remember to git pull to retrieve changes
2. Don't submit massive commits. Gradually and incrementally commit each time a change is made that can alter the code. Large problems arise when teams don't do this.
3. When editing the same files be careful and aware of other users. Github is not like google docs.

#### Staging/Pushing Commits
>`git add <filename>` Stage file for commit
>`git commit -m '<message>` Commit changes
>`git push origin <branch>` Push changes to branch (i.e. master/Carly)

#### Checking History
>`git log` Gives history of previous/recent commits

## Branches
#### Branch Creation
>`git checkout -b <Branch Name>`

#### Move between branches
>`git checkout <Branch Name>`
Note: You cannot move between branches created by other users only the ones created on your local laptop/computer

#### Delete a branch
>`git branch -d <Branch Name>`

#### View all branches
>`git branch`

#### Merging Branches (Simple)
>`git checkout master`
>`git pull origin master`
>`git merge <Branch Name>`
>`git push origin master`

#### Git pull vs Git pull --rebase

>`git pull` Fetches most recent changes to current branch from remote

>`git fetch` `git merge`  Is usually better approach to maintain the code

>`git pull --rebase` Rather than merge changes it pulls changes then adds your local change ontop of the pulled changes

#### Sorting through merge conflicts
Merge conflicts arise when there are multiple versions of the code committed. This raises an error because git itself cannot decide which version, as this is up to the developer

As a consequence, git will add the following to the files with the merge conflict
`<<<<<<< HEAD` Indicates start of the merge conflict for your changes
`=======` indicates the divide between your changes and the changes of the other branch
`>>>>>>> Branch_Name` Indicates end of the merge conflict details for the branch

Delete these markers and write the changes you actually want then git add/commit/push
