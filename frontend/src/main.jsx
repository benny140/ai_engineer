import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { CssBaseline, ThemeProvider, createTheme } from '@mui/material'
import '@fontsource/manrope/400.css'
import '@fontsource/manrope/600.css'
import '@fontsource/manrope/700.css'
import '@fontsource/ibm-plex-mono/400.css'
import './index.css'
import App from './App.jsx'

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#164272',
    },
    secondary: {
      main: '#0d7f8c',
    },
    background: {
      default: '#eef4f8',
      paper: '#ffffff',
    },
    text: {
      primary: '#0f2233',
      secondary: '#51637a',
    },
  },
  shape: {
    borderRadius: 16,
  },
  typography: {
    fontFamily: 'Manrope, Segoe UI, sans-serif',
    h5: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    body2: {
      fontWeight: 500,
    },
    caption: {
      fontFamily: 'IBM Plex Mono, Consolas, monospace',
      fontWeight: 400,
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: 12,
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          borderRadius: 10,
        },
      },
    },
  },
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </StrictMode>,
)
