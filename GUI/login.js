import { initializeApp } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-app.js";
import { getAuth, signInWithPopup, GoogleAuthProvider, signOut } from "https://www.gstatic.com/firebasejs/10.13.1/firebase-auth.js";

// Firebase 配置
const firebaseConfig = {
    apiKey: "AIzaSyCmIOrT1sVMTuR3kC_UF2VikkVa2Ct7gq8",
    authDomain: "tm-project-c29c8.firebaseapp.com",
    databaseURL: "https://tm-project-c29c8-default-rtdb.asia-southeast1.firebasedatabase.app",
    projectId: "tm-project-c29c8",
    storageBucket: "tm-project-c29c8.appspot.com",
    messagingSenderId: "85473383713",
    appId: "1:85473383713:web:811961152070add704da60",
    measurementId: "G-RWGPQZCQZF"
};

// 初始化 Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

// 防止自動登入
auth.signOut();

// 登入按鈕
document.getElementById('loginBtn').addEventListener('click', () => {
    const provider = new GoogleAuthProvider();
    
    // Google 登入
    signInWithPopup(auth, provider)
    .then((result) => {
        // 登入成功
        const user = result.user;
        localStorage.setItem('userId', user.uid); // 將 userId 存入 localStorage
        document.getElementById('loginStatus').textContent = 'success!';
        
        // 兩秒後跳轉到主畫面
        setTimeout(() => {
            window.location.href = 'chat.html';
        }, 500);
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('loginStatus').textContent = 'Error, Please try again.';
    });
});

