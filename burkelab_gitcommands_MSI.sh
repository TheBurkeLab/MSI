# # Move a folder into another folder
# mv <path_to_folder> <path_to_destination>

# # Push to a remote branch
# git push origin <branch_name>

# # Set up a remote tracking branch
# git checkout --track origin/branch_name

# # See the list of local branches:
# Git branch

# # Switch to a specific local branch:
# Git branch <branchname>

# # Create a new local branch according to a specific commit found within a pre-existing branch.
# newBranchName=burkelab_PCI2024_newCode
# commitToCopy=24ac148a9f791c9f72da40e5e3edf2cd4ef89e31
# git checkout -b $newBranchName $commitToCopy

# # Push a new locally created branch to the github repository:
# Git push origin <new_branch>

# # See commit history on a specific branch: (press ‘q’ to quit once done)
# branchName=burkelabNov30_newcode
# # git log $branchName
# git log $branchName --pretty=format:"%h - %an, %ar : %s"

# Delete a branch both locally and remotely:
# 1.	Delete the local branch: 
branchName=new_branch_name
git branch -d $branchName # (if the branch has unmerged changes, you may need to force delet it using ‘-D’ instead of ‘-d’)
# 2.	Delete the remote branch: 
git push origin --delete $branchName

# # Rename a branch both locally and remotely:
# old_branch_name=psingal_MSI
# new_branch_name=psingalMSI
# git checkout $old_branch_name # 1
# git branch -m $new_branch_name # 2
# git push origin -u $new_branch_name # 3
# git push origin --delete $old_branch_name # 4

# # To merge the latest version of the remote github branch and your local branch:
# # 1.	Fetch the latest changes from the remote repository: 
# git fetch origin
# # 2.	Checkout your local branch: 
# git checkout <yourLocalBranch>
# # 3.	Merge the remote branch into your local branch: 
# git merge origin/remote_branch_name

# # To overwrite your local branch with a remote branch:
# # 1.	Fetch the latest changes from the remote repository: 
# git fetch origin
# # 2.	Reset your local branch to the remote branch: 
# git reset –hard origin/remote_branch_name
# # 3.	Force push the changes to the remote repository: 
# git push origin your_local_branch –force

# # Copy a subdirectory from a previous commit on a different branch to the current version:
# # 1.	Check out the target branch: 
# git checkout burkelab
# # 2.	Create a temporary directory to copy the subdirectory from the commit: 
# mkdir temp_directory
# # 3.	Check out the subdirectory from the specific commit to the temporary directory: 
# git checkout b41d530c125e9bb1318d70d6f6fe7e0e38042ddb -- C:\\Users\\pjsin\\Documents\\cantera\\burkelab_SimScripts
# # 4.	Move the subdirectory to the desired location in your current branch: 
# mv temp_directory/burkelab_SimScripts path/to/desired/location
# # 5.	Add the subdirectory to the staging area: 
# git add path/to/desired/location/burkelab_SimScripts
# # 6.	Commit the changes: 
# git commit -m "Add burkelab_SimScripts from commit b41d530c125e9bb1318d70d6f6fe7e0e38042ddb on branch burkelab_PCI2024_oldCode"
# # 7.	Clean up the temporary directory: 
# rm -rf temp_directory

# # Temporarily view a specific past git commit (using the example of 640d7f9):
# git checkout 640d7f9
# git checkout 7716350678b908d7ce3f68e22750349b1f778341 #NOVEMBER 30th VERSION

# # Conda command to install boost:
# conda install boost-cpp 
