
# Dream11 Product Setup Guide

This guide provides detailed instructions for setting up the Dream11 Product UI, Model UI, and the backend services required to run the application.

---

## Installation and Setup Instructions

### Prerequisites

Before proceeding, ensure the following are installed on your system:

- **Node.js** (v14.x or later): [Download and install Node.js](https://nodejs.org/).  
- **npm**: Comes pre-installed with Node.js. Verify by running `npm -v` in your terminal.  
- A terminal/command prompt.

---

### Step 1: Download the Project

1. Download the provided `.zip` file containing the project source code.
2. Extract the contents of the `.zip` file to a directory of your choice.  

---

### Step 2: Install Dependencies

The project contains three main components: Product UI, Backend, and Model UI. Each requires separate dependency installation.

#### a. Install Dependencies for Product UI  
Navigate to the `dream11_frontend` folder:  
```bash
cd ui/dream11_frontend
npm install
```

#### b. Install Dependencies for Backend  
Navigate to the `dream11_backend` folder:  
```bash
cd ../dream11_backend
npm install
```

#### c. Install Dependencies for Model UI  
Navigate to the `modelUI_frontend` folder:  
```bash
cd ../modelUI_frontend
npm install
```

---

### Step 3: Start the Services

Start each service by running the following commands in separate terminal windows:

#### a. Start the Product UI  
From the `dream11_frontend` folder:  
```bash
cd ui/dream11_frontend
npm run dev
```

#### b. Start the Backend  
From the `dream11_backend` folder:  
```bash
cd ../dream11_backend
npm run dev
```

#### c. Start the Model UI  
From the `modelUI_frontend` folder:  
```bash
cd ../modelUI_frontend
npm run dev
```

---

### Step 4: Access the Application

1. Open your web browser.  
2. Navigate to `http://localhost:<PORT>` (the port displayed in the terminal for each service).  

   Example:  
   - Product UI: `http://localhost:3000`  
   - Backend: Refer to the port displayed in its terminal.  
   - Model UI: Refer to the port displayed in its terminal.

---

Your application is now up and running! 🎉
