# Template Repo for ML Project

This template repo will give you a good starting point for your second project. Besides the files used for creating a virtual environment, you will find a simple example of how to build a simple model in a python script. This is maybe the simplest way to do it. We train a simple model in the jupyter notebook, where we select only some features and do minimal cleaning. The output is then stored in simple python scripts.

The data used for this is: [coffee quality dataset](https://github.com/jldbc/coffee-quality-database).

---

## Set up a Kanban board on github

Go to ML-Project Template.

Click on "Use this Template" (Green button)
![alt text](./images/step_1a.png)

Create new project with relevant name, the owner should be your own account and **not** Neuefische. 

![alt text](./images/step_2.png)

In your newly create repo, navigate to "Projects", and then click on "Add project" (green button). Normally you don't have created a project yet, so you can click the arrow navigation to create project on your profile. This project can be added at the end to your repository.
![alt text](./images/add_project.png)


You will be guided to your profiles projects. Click here on the green button "New project" to create a new project. Choose "board" view and **not** "table" view, then click "create".
Good, now you have a board view. 
![alt text](./images/boardview.png)

Now change the name of your board, to match that of your chosen ML project. 

Next, assign rights to all your team members by clicking on the 3 dots on the top right of the board, and then go to "settings". 

Next, click on "Manage Access"
Add your team mates by Searching for their github handle in the search window.

Change their Role from ‘View’ to ‘Admin’. 
Click on the green button “Invite” to add them. Repeat for all team members.
![alt text](./images/team_access.png
)

Next, add action items with the relevant name e.g. “load data”, "get statistics", etc.

Convert added item to issue by clicking on the 3 dots on the particular added item.

Then select the repo you created in step3 for the issue to be added. (Select the project repo example “Fraud detection”)



When in project repo, Go to issues, then go to milestones. 

Click on ”Add milestone”.

Give the milestone a due date and description as per the example provided by the coaches. 

Add description of: 

A) What needs to be completed to be done with the milestone

B) The definition of done: what will your result look like when you have completed the milestone? (check the provided format)
![alt text](./images/create_milestone.png)

Now navigate to "issues".

Assign issues to milestones, give it assignees (people who will work on the task). 
![alt text](./images/tasks_to_mileston.png)

### Optional: Add workflows

Workflows can help you keep your kanban board automatically on track. 

Select the project created in the steps above.  

Click on the 3 dots to the far right of the board (...)

Select workflow as the first option. 

Activate the ones you feel necessary to your project

Go back to your project repository (fraud detection))

## Requirements and Environment

Requirements:
- pyenv with Python: 3.11.3

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python example_files/train.py  
```

In order to test that predict works on a test set you created run:

```bash
python example_files/predict.py models/linear_regression_model.sav data/X_test.csv data/y_test.csv
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.


