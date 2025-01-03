import React, { useState } from "react";
import { useNavigate } from "react-router-dom";

// Helper function to validate and parse dd-mm-yyyy format
const parseDate = (dateString) => {
  const regex = /^(\d{2})-(\d{2})-(\d{4})$/; // Match dd-mm-yyyy format
  const match = dateString.match(regex);
  if (!match) return null; // Invalid format
  const [, day, month, year] = match;
  const parsedDate = new Date(`${year}-${month}-${day}`);
  return isNaN(parsedDate) ? null : parsedDate;
};

const DateInputForm = () => {
  const [dates, setDates] = useState({
    train_start_date: "",
    train_end_date: "",
    test_start_date: "",
    test_end_date: "",
    format: "ODI", // Default format
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setDates({ ...dates, [e.target.name]: e.target.value });
    setError("");
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const { train_start_date, train_end_date, test_start_date, test_end_date, format } = dates;

    // Validate date inputs
    const parsedTrainStart = parseDate(train_start_date);
    const parsedTrainEnd = parseDate(train_end_date);
    const parsedTestStart = parseDate(test_start_date);
    const parsedTestEnd = parseDate(test_end_date);

    if (!parsedTrainStart || !parsedTrainEnd || !parsedTestStart || !parsedTestEnd) {
      setError("Please enter valid dates in dd-mm-yyyy format.");
      return;
    }

    if (parsedTrainStart > parsedTrainEnd) {
      setError("Training start date must be before the end date.");
      return;
    }
    if (parsedTestStart > parsedTestEnd) {
      setError("Testing start date must be before the end date.");
      return;
    }

    try {
      setLoading(true);

      // Dates are already in dd-mm-yyyy format, no need for extra formatting
      const formattedDates = {
        train_start_date,
        train_end_date,
        test_start_date,
        test_end_date,
        format,
      };

      console.log("Formatted Dates:", formattedDates);

      const response = await fetch(`${import.meta.env.VITE_BACKEND}/train`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formattedDates),
      });

      if (!response.ok) {
        throw new Error("Failed to train. Please try again.");
      }

      const result = await response.text(); // Use response.text() for CSV
      console.log("Training Results:", result);

      navigate("/results", { state: { result } });
    } catch (err) {
      console.error(err);
      setError(err.message || "An error occurred. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-red-500 to-red-700">
      <div className="bg-white shadow-2xl rounded-3xl w-full max-w-4xl p-10">
        <h1 className="text-4xl font-bold text-gray-800 mb-8 text-center">
          Schedule Training & Testing
        </h1>

        {error && <div className="bg-red-100 text-red-600 p-4 rounded mb-4">{error}</div>}

        {loading ? (
          <div className="loader border-t-4 border-red-600 rounded-full w-12 h-12 animate-spin mx-auto"></div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="grid grid-cols-2 gap-6">
              {["train_start_date", "train_end_date", "test_start_date", "test_end_date"].map(
                (field) => (
                  <div key={field}>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {field.split(/_/).join(" ").replace(/\b\w/g, (l) => l.toUpperCase())} (dd-mm-yyyy)
                    </label>
                    <input
                      type="text"
                      name={field}
                      placeholder="dd-mm-yyyy"
                      value={dates[field]}
                      onChange={handleChange}
                      className="w-full p-3 border rounded"
                    />
                  </div>
                )
              )}

              {/* Dropdown for Format */}
              {/* <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Format</label>
                <select
                  name="format"
                  value={dates.format}
                  onChange={handleChange}
                  className="w-full p-3 border rounded bg-white"
                >
                  <option value="ODI">ODI</option>
                  <option value="T20">T20</option>
                  <option value="Test">Test</option>
                </select>
              </div> */}
            </div>

            <button
              type="submit"
              className="mt-6 w-full bg-red-500 text-white p-3 rounded shadow hover:bg-red-600"
            >
              Submit Dates
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default DateInputForm;
