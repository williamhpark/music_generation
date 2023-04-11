# Music Generating Recurrent Neural Network

### Summary

Music has always been a huge part of my life. I've played the guitar since I was 11 year old, trumpet throughout all of high school, and just recently I started exploring music production.

My recent goal of learning to produce music using the Ableton DAW has been a lot of fun, but also very frustrating, especially when I'm trying to think of melodies to base the songs off of. This made me consider how machine learning can be used to generate melodies, which I could maybe use for inspiration when producing, and was thus the source of inspiration for this project.

### Demonstration

To see how the RNN model pipeline was built, refer to [this Colab notebook](https://github.com/williamhpark/music_generation/blob/main/Music_Generating_Recurrent_Neural_Network.ipynb), which can be ran in Google Colab.

Here's a demonstration of the web UI's functionality (sound on). The model currently requires more tuning to make the melodies sound more "musical":

https://user-images.githubusercontent.com/53918631/191257297-5bcafe82-c7ae-47cf-b178-320d4d956b62.mov

### Technologies

- TensorFlow
- Flask

### Next Steps

- Fix the live URL (currently undergoing maintenance)
- Improve the frontend UI, possibly with the help of a frontend library (e.g. React)
- Experiment with tuning the parameters (i.e. loss weights, etc) for lower total loss and better "musicality"
- Deploy and train the model on a cloud service (e.g. Amazon SageMaker)
- Allow the user to customize the melody, e.g. length, speed
- Allow the user to generate melodies based on a selected genre (this would require the use of a different dataset where genre metadata is available)
- Explore using GANs for music generation
