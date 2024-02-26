library(shiny)
library(png)

# Define the user interface
ui <- fluidPage(
  titlePanel("Diagnostic Test Calculator"),  # Title of the application
  sidebarLayout(
    sidebarPanel(
      selectInput("test_type", "Select Test Type:",  # Dropdown to select test type
                  choices = c("Eliza", "RCR", "Culture")),
      numericInput("positive_cases", "Number of Positive Cases:", value = 0),  # Input for positive cases
      numericInput("negative_cases", "Number of Negative Cases:", value = 0),  # Input for negative cases
      actionButton("continue_button", "Continue")  # Continue button
    ),
    mainPanel(
      textOutput("total_tested"),  # Output for total tested
      tags$br(),  # Line break
      tags$img(src = "goat.png", height = "200px", width = "300px")  # Display the image
      
    )
  )
)

# Define the server logic
server <- function(input, output) {
  output$total_tested <- renderText({
    # Calculate the total number tested
    total <- input$positive_cases + input$negative_cases
    paste("Total Number Tested:", total)  # Display the total number tested
  })
  
  observeEvent(input$continue_button, {
    # Add the code for what happens when the continue button is clicked
    # You can add more outputs or actions here
  })
}

# Run the Shiny app
shinyApp(ui = ui, server = server)