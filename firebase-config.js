// Replace these values with your Firebase project configuration
firebaseConfig = {
    apiKey: "AIzaSyD6ZsHhBNw5m9PFIE7sVUkwWsaaOQRLDuQ",
    authDomain: "cv-align.firebaseapp.com",
    projectId: "cv-align",
    storageBucket: "cv-align.firebasestorage.app",
    messagingSenderId: "189147681902",
    appId: "1:189147681902:web:1a46a5f99f536a292ac27f",
    measurementId: "G-C6363C30KN"
  };

// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const storage = firebase.storage();

// Export the storage instance for use in other files
window.firebaseStorage = storage; 