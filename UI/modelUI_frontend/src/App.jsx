import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DateInputForm from "./DateInputForm";
import Results from "./Results";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<DateInputForm />} />
        <Route path="/results" element={<Results />} />
      </Routes>
    </Router>
  );
}

export default App;
