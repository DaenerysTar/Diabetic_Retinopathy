import { createRouter, createWebHistory } from 'vue-router'
import HomePage from '../views/HomePage.vue'
import PredictionPage from '../components/PredictionPage.vue'
import ResultsPage from '../views/ResultsPage.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: HomePage
  },
  {
    path: '/prediction',
    name: 'Prediction',
    component: PredictionPage
  },
  {
    path: '/results',
    name: 'Results',
    component: ResultsPage
  }
]

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes
})

export default router
