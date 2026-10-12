import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { BuildFooter } from './components/BuildFooter'
import { JobPage } from './pages/JobPage'
import { LaunchPage } from './pages/LaunchPage'

export default function App() {
  return (
    <BrowserRouter>
      <div className="app-shell">
        <div className="app-main">
          <Routes>
            <Route path="/" element={<LaunchPage />} />
            <Route path="/jobs/:id" element={<JobPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
        <BuildFooter />
      </div>
    </BrowserRouter>
  )
}
