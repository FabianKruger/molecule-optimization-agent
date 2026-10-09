import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { JobPage } from './pages/JobPage'
import { LaunchPage } from './pages/LaunchPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LaunchPage />} />
        <Route path="/jobs/:id" element={<JobPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
