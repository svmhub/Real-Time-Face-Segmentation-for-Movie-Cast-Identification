# Real-Time-Face-Segmentation-for-Movie-Cast-Identification
Scene Cast AI: Real-Time Face Segmentation for Movie Cast Identification. Company X's streaming app needs to automatically detect and segment faces in movie scene screenshots so users can pause videos and instantly view cast/crew details for actors on screen.

### Business Domain Value

   - Pause-and-identify feature: Users pause movies to see actor names/profiles overlayed on
detected faces.
   - Personalized recommendations: Track viewer preferences for specific actors across films.
   - Content moderation: Automatically flag inappropriate scenes via face detection.
   - Advertising: Dynamic ads featuring favourite actors during streaming breaks.

### The approach has been followed

   1. The Pre-processing has been done by externally.
      - Tain data has been provided in .npy processed file. It is size in huge, so could not upload it in the repository.
      - Test data also provided here.
   2. Data understanding, Data Visualization, EDA:
      - Load dataset of movie scene images and corresponding binary face masks.
      - EDA and Data visualization has been done and shown.
      - Preprocess the data: Resize to 256x256,
      - Data Augmentation:
           - Augment (flip/rotate) image for varied poses/lighting/blurring/sharpening.
   3. Model Building:
      - Built U-Net model with MobileNetV2 encoder (transfer learning), custom decoder with skip connections.
      - Trained the mode
      - Implemented custom Dice Coefficient metric and Dice Loss function.
      - To dealt with large training time, saved the weights so that you can use them when training the model for the second time without starting from scratch.
   4. Test the Model, Fine-tuning and Repeat:
      - Tested the model with previous one and reported as per evaluation metrics
      - Tried in different hyper parameters in the same models
      - Set different hyper parameters, by trying different optimizers, loss functions, epochs, learning rate, batch size, checkpointing, early stopping etc. for these models to fine-tune them
      - Reported evaluation metrics for these models along with your observation on how changing different hyper parameters leads to change in the final evaluation metric
   5. Deployment:
      - Integrated the trained model into a Streamlit-based web application.
      - Provided an interface where users can upload images or use real-time webcam feeds for detection.

***Note: I have done this project in the .ipynb file on VS Code. And uploaded the same file in the repository. I mentioned detailed explanation of each and every step of this project here itself. Kindly look at it, then you can understand the things in this project***

## Hence the project has been done successfully by me! 🤞😂😍
