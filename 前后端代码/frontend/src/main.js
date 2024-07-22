
import ElementPlus from 'element-plus';
import 'element-plus/theme-chalk/index.css';
import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js'; //引入定义好的路由
createApp(App).use(ElementPlus).use(router).mount('#app') //使用路由

