Instructions to get the Unity environment set up and running.

1. Download the Unity Hub from https://unity.com/download
2. Once unity is installed, sign in with your account
3. Install editor version 6000.0.277f1
4. Select "Universal 3d" as your project type and name it as you wish.
5. Once you are inside the project, navigate to the project folder
6. Drag the assets, packages and project settings from github and copy them to the unity folder
7. Done.

To run training:
1. Have VSCode and Unity open at the same time
2. Navigate to the train.py (or old_train.py) and click the run button in vscode.
3. Quickly swap to unity and play the unity editor (triangle play button at the top)
4. Watch training work!

To run inference:
1. Have a valid ML model and put it into your assets folder
2. Select all agents in the unity hierarchy 
3. In the gameobject settings, set the model from "None" to the model you have in the assets folder
4. set behavior type to "inference only"
5. Run the unity scene using the play button