import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";

const Results = () => {
  const location = useLocation();
  const { result } = location.state || {};
  const [parsedCSV, setParsedCSV] = useState(null);

  useEffect(() => {
    if (result) {
      try {
        const parsed = parseCSV(result);
        setParsedCSV(parsed);
      } catch (error) {
        console.error("Error parsing CSV:", error);
      }
    }
  }, [result]);

  const parseCSV = (csv) => {
    const rows = csv.split("\n").map((row) => row.split(","));
    const headers = rows[0];
    const data = rows.slice(1).filter((row) => row.length === headers.length); // Filter out incomplete rows
    return { headers, data };
  };

  const downloadCSV = () => {
    const blob = new Blob([result], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "training_results.csv";
    link.click();
    URL.revokeObjectURL(url);
  };

  if (!result) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-red-500 to-red-700">
        <p className="text-white text-lg">No results available.</p>
      </div>
    );
  }

  if (!parsedCSV) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-red-500 to-red-700">
        <p className="text-white text-lg">Loading CSV...</p>
      </div>
    );
  }

  const { headers, data } = parsedCSV;

  return (
    <div className="flex items-center justify-center min-h-screen bg-gradient-to-r from-red-500 to-red-700 p-6">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl p-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800">Training Results</h1>
          <button
            onClick={downloadCSV}
            className="bg-red-500 text-white px-6 py-3 rounded-lg shadow hover:bg-red-600 transition duration-200"
          >
            Download CSV
          </button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full table-auto border-collapse border border-gray-200 rounded-lg">
            <thead>
              <tr>
                {headers.map((header, idx) => (
                  <th
                    key={idx}
                    className="px-4 py-3 border border-gray-300 bg-red-100 text-gray-700 font-semibold text-sm text-left"
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, rowIdx) => (
                <tr
                  key={rowIdx}
                  className={`${
                    rowIdx % 2 === 0 ? "bg-red-50" : "bg-white"
                  } hover:bg-red-100 transition duration-200`}
                >
                  {row.map((cell, cellIdx) => (
                    <td
                      key={cellIdx}
                      className="px-4 py-2 border border-gray-300 text-gray-600 text-sm"
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default Results;
