import React from "react";
import Header from "./components/Header/Header";
import Footer from "./components/Footer/Footer";
import { Outlet } from "react-router-dom";
import { ThemeProvider } from "./ThemeContext";

function Layouts() {
  return (
    <ThemeProvider>
      <Header />
      <div className="mt-6 mb-6 px-1">
        <Outlet />
      </div>
      <Footer />
    </ThemeProvider>
  );
}

export default Layouts;
