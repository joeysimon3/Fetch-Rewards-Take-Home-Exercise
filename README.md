# Fetch-Rewards-Take-Home-Exercise
Machine learning exercise for Fetch Rewards


Script is run with Python3 and uses Tensorflow for the AI model and Tkinter for the example GUI.

# Start up instructions
  1) Set up a new python virtual environment by entering this line into the terminal: 

  ```shell
  pip install virtualenv 
  virtualenv venv
  source venv/bin/activate
  ```  
  You can simply enter ```deactivate``` to exit the virtual environment.

  2) Execute the dependencies file:

  ```shell
  pip install  -r requirements.txt
  ```

  3) Execute the model build which builds the model and estimates receipts for the next year:

  ```shell
  python3 Code/buildModel.py
  ```

  4) Start up the GUI which allows you to query the sums from any two date ranges of the predicted receipts:

  ```shell
  python3 Gui/gui.py
  ```

